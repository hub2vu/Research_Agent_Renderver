/**
 * SidePanel Component
 *
 * Displays detailed information about a selected paper.
 * Used in both GlobalGraphPage and PaperGraphPage.
 *
 * [UPDATED]
 * - Node color override uses stableKey (if exists) for persistence
 * - arXiv link robust: tries node.id first, then node.title
 * - Keep backward compatibility with existing props
 */

import React, { useMemo } from 'react';
import { GraphNode } from '../lib/mcp';
import PaperCard from './PaperCard';

/* ----------------------- Helper: stable key ---------------------- */

function nodeKeyOf(node: any): string {
  return String(node?.stableKey ?? node?.id ?? '');
}

/* ----------------------- Helper: arXiv link ---------------------- */

function extractArxivId(raw: string): string | null {
  if (!raw) return null;
  let s = String(raw).trim();

  // URL 형태면 ID만 추출
  const urlMatch = s.match(/arxiv\.org\/(?:abs|pdf)\/([^?#]+?)(?:\.pdf)?/i);
  if (urlMatch?.[1]) s = urlMatch[1];

  // 접두 제거
  s = s.replace(/^arxiv:\s*/i, '');

  // DOI-like: 10.48550_arxiv.<id>
  s = s.replace(/^10\.48550[_\/]arxiv\./i, '');

  // 구형 ID underscore 포맷 정규화
  // cs_AI_0112017v1 -> cs.AI/0112017v1
  if (/^[a-z\-]+_[A-Z]{2}_\d{7}(v\d+)?$/.test(s)) {
    const parts = s.split('_');
    if (parts.length >= 3) s = `${parts[0]}.${parts[1]}/${parts.slice(2).join('_')}`;
  } else if (/^[a-z\-]+_[A-Z]{2}\/\d{7}(v\d+)?$/.test(s)) {
    const i = s.indexOf('_');
    s = s.slice(0, i) + '.' + s.slice(i + 1);
  }

  const modern = /^\d{4}\.\d{4,5}(v\d+)?$/;          // 2506.07976v2
  const old = /^[a-z\-]+\.[A-Z]{2}\/\d{7}(v\d+)?$/; // cs.AI/0112017v1

  if (modern.test(s) || old.test(s)) return s;
  return null;
}

function getArxivAbsUrlFromNode(node: any): string | null {
  // 1) id에서 먼저 찾고
  const byId = extractArxivId(String(node?.id ?? ''));
  if (byId) return `https://arxiv.org/abs/${byId}`;

  // 2) title에서도 찾는다 (ref:* 노드 대비)
  const byTitle = extractArxivId(String(node?.title ?? ''));
  if (byTitle) return `https://arxiv.org/abs/${byTitle}`;

  return null;
}

/* ---------------------- Helper: color ---------------------- */

const PRESET_COLORS = [
  '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
  '#f56565', '#38b2ac', '#ed64a6', '#667eea',
  '#1a202c', '#718096'
];

function isHexColor(v: unknown): v is string {
  return typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v);
}

/* ----------------------------- Props ----------------------------- */

interface SidePanelProps {
  selectedNode: GraphNode | null;
  onClose: () => void;

  // 기존 페이지에서 쓰는 액션(있을 수도/없을 수도)
  onExpand?: (node: GraphNode) => void;
  onNavigate?: (node: GraphNode) => void;

  // 페이지에서 넘기고 있을 수 있어서 남겨둠(사용 안 함)
  onAction?: () => void;

  mode: 'global' | 'paper';
  isLoading?: boolean;

  // nodeKey(stableKey 우선) -> color
  nodeColorMap?: Record<string, string>;
  nodeColor?: string;
  onNodeColorChange?: (nodeKey: string, color: string) => void;
  onNodeColorReset?: (nodeKey: string) => void;
}

export default function SidePanel({
  selectedNode,
  onClose,
  onExpand,
  onNavigate,
  mode,
  isLoading = false,
  nodeColorMap,
  nodeColor,
  onNodeColorChange,
  onNodeColorReset
}: SidePanelProps) {
  if (!selectedNode) return null;

  const selectedIsCenter = Boolean((selectedNode as any).is_center || (selectedNode as any).isCenter);
  const arxivUrl = getArxivAbsUrlFromNode(selectedNode);

  // ✅ stableKey 우선 키
  const selectedKey = useMemo(() => nodeKeyOf(selectedNode as any), [selectedNode]);

  const currentColor = useMemo(() => {
    // 1) 상위에서 직접 주는 값이 있으면 그거 우선
    if (isHexColor(nodeColor)) return nodeColor;

    // 2) stableKey 기반 조회
    const byKey = nodeColorMap?.[selectedKey];
    if (isHexColor(byKey)) return byKey;

    // 3) backward compat: 예전엔 id로 저장했을 수 있음
    const byId = nodeColorMap?.[selectedNode.id];
    if (isHexColor(byId)) return byId;

    // 4) default
    return '#4299e1';
  }, [nodeColor, nodeColorMap, selectedKey, selectedNode.id]);

  const canEditColor = Boolean(onNodeColorChange || onNodeColorReset);

  const emitColorChange = (color: string) => {
    if (!onNodeColorChange) return;
    if (!isHexColor(color)) return;
    onNodeColorChange(selectedKey, color);
  };

  const emitColorReset = () => {
    if (!onNodeColorReset) return;
    onNodeColorReset(selectedKey);
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: '320px',
        height: '100vh',
        backgroundColor: '#fff',
        borderLeft: '1px solid #e2e8f0',
        boxShadow: '-4px 0 12px rgba(0, 0, 0, 0.08)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 9999
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '16px',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <h3 style={{ margin: 0, fontSize: '14px', color: '#4a5568' }}>Paper Details</h3>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '18px',
            cursor: 'pointer',
            color: '#a0aec0',
            lineHeight: 1
          }}
          aria-label="Close"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div style={{ padding: '16px', overflowY: 'auto', flex: 1 }}>
        <PaperCard node={selectedNode} />

        {/* -------- Node Color Controls -------- */}
        <div
          style={{
            marginTop: '14px',
            paddingTop: '14px',
            borderTop: '1px solid #edf2f7'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div style={{ fontSize: '12px', color: '#718096', fontWeight: 600 }}>
              Node color
            </div>
            {!canEditColor && (
              <div style={{ fontSize: '11px', color: '#a0aec0' }}>
                (connect callbacks)
              </div>
            )}
          </div>

          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="color"
              value={currentColor}
              disabled={!onNodeColorChange}
              onChange={(e) => emitColorChange(e.target.value)}
              style={{
                width: 42,
                height: 34,
                padding: 0,
                border: 'none',
                background: 'transparent',
                cursor: onNodeColorChange ? 'pointer' : 'not-allowed'
              }}
              aria-label="Pick node color"
            />

            <input
              type="text"
              value={currentColor}
              disabled={!onNodeColorChange}
              onChange={(e) => {
                const v = e.target.value.trim();
                // 타이핑 중엔 무시, 유효한 hex일 때만 반영
                if (isHexColor(v)) emitColorChange(v);
              }}
              style={{
                flex: 1,
                padding: '8px 10px',
                border: '1px solid #e2e8f0',
                borderRadius: 6,
                fontSize: 12,
                color: '#2d3748'
              }}
              placeholder="#RRGGBB"
            />

            <button
              disabled={!onNodeColorReset}
              onClick={emitColorReset}
              style={{
                padding: '8px 10px',
                borderRadius: 6,
                border: '1px solid #e2e8f0',
                background: '#f7fafc',
                cursor: onNodeColorReset ? 'pointer' : 'not-allowed',
                fontSize: 12,
                color: '#4a5568',
                opacity: onNodeColorReset ? 1 : 0.6
              }}
            >
              Reset
            </button>
          </div>

          <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
            {PRESET_COLORS.map((c) => (
              <button
                key={c}
                disabled={!onNodeColorChange}
                onClick={() => emitColorChange(c)}
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 999,
                  border:
                    c.toLowerCase() === currentColor.toLowerCase()
                      ? '2px solid #2d3748'
                      : '1px solid #e2e8f0',
                  background: c,
                  cursor: onNodeColorChange ? 'pointer' : 'not-allowed',
                  opacity: onNodeColorChange ? 1 : 0.6
                }}
                aria-label={`Set color ${c}`}
                title={c}
              />
            ))}
          </div>

          <div style={{ marginTop: 8, fontSize: 11, color: '#a0aec0' }}>
            Tip: 선택 하이라이트는 테두리로 표시돼서, 사용자 지정 색은 그대로 유지돼.
          </div>

          <div style={{ marginTop: 6, fontSize: 11, color: '#a0aec0' }}>
            Key: <code>{selectedKey}</code>
          </div>
        </div>

        {/* -------- Actions -------- */}
        <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {mode === 'global' && onNavigate && (
            <button
              onClick={() => onNavigate(selectedNode)}
              disabled={isLoading}
              style={{
                padding: '10px 16px',
                backgroundColor: '#4299e1',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.7 : 1
              }}
            >
              View Reference Graph →
            </button>
          )}

          {mode === 'paper' && onExpand && !selectedIsCenter && (
            <button
              onClick={() => onExpand(selectedNode)}
              disabled={isLoading}
              style={{
                padding: '10px 16px',
                backgroundColor: '#48bb78',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.7 : 1
              }}
            >
              {isLoading ? 'Expanding...' : 'Expand References'}
            </button>
          )}

          {arxivUrl && (
            <a
              href={arxivUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                padding: '10px 16px',
                backgroundColor: '#f7fafc',
                color: '#4a5568',
                border: '1px solid #e2e8f0',
                borderRadius: '6px',
                textAlign: 'center',
                textDecoration: 'none'
              }}
            >
              View on arXiv ↗
            </a>
          )}
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid #e2e8f0',
            fontSize: '12px',
            color: '#2b6cb0',
            backgroundColor: '#ebf8ff'
          }}
        >
          Processing...
        </div>
      )}
    </div>
  );
}
