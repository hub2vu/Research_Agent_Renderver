/**
 * SidePanel Component
 *
 * Displays detailed information about a selected paper.
 * Used in both GlobalGraphPage and PaperGraphPage.
 */

import React from 'react';
import { GraphNode } from '../lib/mcp';
import PaperCard from './PaperCard';

/* ----------------------- Helper: arXiv link ---------------------- */

function extractArxivId(rawId: string): string | null {
  if (!rawId) return null;
  let s = String(rawId).trim();

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

function getArxivAbsUrl(nodeId: string): string | null {
  const arxivId = extractArxivId(nodeId);
  return arxivId ? `https://arxiv.org/abs/${arxivId}` : null;
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
}

export default function SidePanel({
  selectedNode,
  onClose,
  onExpand,
  onNavigate,
  mode,
  isLoading = false
}: SidePanelProps) {
  if (!selectedNode) return null;

  const selectedIsCenter = Boolean((selectedNode as any).is_center || (selectedNode as any).isCenter);
  const arxivUrl = getArxivAbsUrl(selectedNode.id);

  return (
    <div
      style={{
        position: 'fixed',        // ✅ 항상 화면 오른쪽에 고정
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
        {/* ✅ FIX: PaperCard는 prop 이름이 node */}
        <PaperCard node={selectedNode} />

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

          {/* ✅ arXiv ID가 나올 때만 표시 */}
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

