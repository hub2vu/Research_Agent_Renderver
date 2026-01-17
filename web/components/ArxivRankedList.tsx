/**
 * ArxivRankedList Component
 *
 * Displays ranked search results for arXiv papers.
 */

import React from 'react';
import { ScoredPaper } from './PaperResultCard';

interface ArxivRankedListProps {
  papers: ScoredPaper[];
  onPaperClick: (paperId: string) => void;
}

export default function ArxivRankedList({
  papers,
  onPaperClick,
}: ArxivRankedListProps) {
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
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      padding: '16px',
      borderRadius: '8px',
      width: '400px',
      maxHeight: 'calc(100vh - 200px)',
      overflowY: 'auto',
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    }}>
      <h3 style={{
        margin: '0 0 12px 0',
        color: '#1a202c',
        fontSize: '16px',
        fontWeight: 600,
      }}>
        Ranked Results ({papers.length})
      </h3>

      {papers.map((paper) => {
        const reasoning = (paper as any).reasoning || null;

        return (
          <div
            key={paper.paper_id}
            onClick={() => onPaperClick(paper.paper_id)}
            style={{
              backgroundColor: '#f7fafc',
              borderRadius: '6px',
              padding: '12px',
              marginBottom: '10px',
              border: '1px solid #e2e8f0',
              cursor: 'pointer',
              transition: 'border-color 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#4299e1';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#e2e8f0';
            }}
          >
            {/* Rank and Score */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span style={{
                backgroundColor: '#4299e1',
                color: '#fff',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 600,
              }}>
                #{paper.rank}
              </span>
              <span style={{
                backgroundColor: '#e2e8f0',
                color: '#4a5568',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 500,
              }}>
                Score: {formatScore(paper.score.final)}%
              </span>
            </div>

            {/* Title */}
            <h4 style={{
              margin: '0 0 6px 0',
              color: '#1a202c',
              fontSize: '14px',
              fontWeight: 600,
              lineHeight: 1.4,
            }}>
              {paper.title}
            </h4>

            {/* Authors */}
            <div style={{
              color: '#718096',
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
                backgroundColor: '#edf2f7',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#4a5568',
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
