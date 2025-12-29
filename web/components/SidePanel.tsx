/**
 * SidePanel Component
 *
 * Displays detailed information about a selected paper.
 * Used in both Graph A and Graph B views.
 */

import React from 'react';
import { GraphNode } from '../api/mcp';
import PaperCard from './PaperCard';

interface SidePanelProps {
  selectedNode: GraphNode | null;
  onClose: () => void;
  onExpand?: (node: GraphNode) => void;
  onNavigate?: (node: GraphNode) => void;
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

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        width: '350px',
        height: '100%',
        backgroundColor: '#fff',
        borderLeft: '1px solid #e2e8f0',
        boxShadow: '-4px 0 6px rgba(0, 0, 0, 0.05)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 100
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
        <h3 style={{ margin: 0, fontSize: '14px', color: '#4a5568' }}>
          Paper Details
        </h3>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '20px',
            color: '#a0aec0'
          }}
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        <PaperCard node={selectedNode} />

        {/* Actions */}
        <div
          style={{
            marginTop: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}
        >
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

          {mode === 'paper' && onExpand && !selectedNode.is_center && (
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

          <a
            href={`https://arxiv.org/abs/${selectedNode.id.replace('_', '/')}`}
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
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div
          style={{
            position: 'absolute',
            bottom: '16px',
            left: '16px',
            right: '16px',
            padding: '8px',
            backgroundColor: '#ebf8ff',
            borderRadius: '6px',
            textAlign: 'center',
            fontSize: '12px',
            color: '#2b6cb0'
          }}
        >
          Processing...
        </div>
      )}
    </div>
  );
}
