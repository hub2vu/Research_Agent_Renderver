/**
 * NeurIPSRankedList Component
 *
 * Displays ranked search results for NeurIPS papers with cluster badges and reasoning.
 */

import React from 'react';
import { ScoredPaper } from './PaperResultCard';

interface NeurIPSRankedListProps {
  papers: ScoredPaper[];
  onPaperClick: (paperId: string) => void;
  clusterMap?: Record<string, number>;
}

// Cluster color function (same as PaperCard)
function getClusterColor(cluster: number): string {
  const colors = [
    '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
    '#f56565', '#38b2ac', '#ed64a6', '#667eea'
  ];
  return colors[cluster % colors.length];
}

export default function NeurIPSRankedList({
  papers,
  onPaperClick,
  clusterMap = {},
}: NeurIPSRankedListProps) {
  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  if (papers.length === 0) {
    return null;
  }

  return (
    <div style={{
      position: 'absolute',
      bottom: '16px',
      right: '16px',
      zIndex: 5,
      backgroundColor: 'rgba(26, 32, 44, 0.95)',
      padding: '16px',
      borderRadius: '8px',
      width: '400px',
      maxHeight: 'calc(100vh - 200px)',
      overflowY: 'auto',
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    }}>
      <h3 style={{
        margin: '0 0 12px 0',
        color: '#fff',
        fontSize: '16px',
        fontWeight: 600,
      }}>
        Ranked Results ({papers.length})
      </h3>

      {papers.map((paper) => {
        const clusterId = clusterMap[paper.paper_id] ?? null;
        const reasoning = (paper as any).reasoning || null;

        return (
          <div
            key={paper.paper_id}
            onClick={() => onPaperClick(paper.paper_id)}
            style={{
              backgroundColor: '#1a202c',
              borderRadius: '6px',
              padding: '12px',
              marginBottom: '10px',
              border: '1px solid #2d3748',
              cursor: 'pointer',
              transition: 'border-color 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#4a90d9';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#2d3748';
            }}
          >
            {/* Rank and Score */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span style={{
                backgroundColor: '#4a90d9',
                color: '#fff',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 600,
              }}>
                #{paper.rank}
              </span>
              <span style={{
                backgroundColor: '#2d3748',
                color: '#e2e8f0',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 500,
              }}>
                Score: {formatScore(paper.score.final)}%
              </span>
              {clusterId !== null && (
                <span style={{
                  backgroundColor: getClusterColor(clusterId),
                  color: '#fff',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 500,
                }}>
                  Cluster {clusterId}
                </span>
              )}
            </div>

            {/* Title */}
            <h4 style={{
              margin: '0 0 6px 0',
              color: '#fff',
              fontSize: '14px',
              fontWeight: 600,
              lineHeight: 1.4,
            }}>
              {paper.title}
            </h4>

            {/* Authors */}
            <div style={{
              color: '#a0aec0',
              fontSize: '12px',
              marginBottom: '8px',
            }}>
              {paper.authors.slice(0, 3).join(', ')}
              {paper.authors.length > 3 && ' ...'}
            </div>

            {/* Reasoning */}
            {reasoning && (
              <div style={{
                marginTop: '8px',
                padding: '8px',
                backgroundColor: '#2d3748',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#a0aec0',
                fontStyle: 'italic',
              }}>
                💡 {reasoning}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
