/**
 * ArxivRankedList Component
 *
 * Displays ranked search results for arXiv papers with full details.
 */

import React, { useState, useRef, useEffect } from 'react';
import { ScoredPaper } from './PaperResultCard';

interface ArxivRankedListProps {
  papers: ScoredPaper[];
  onPaperClick: (paperId: string) => void;
  onClose?: () => void;
}

export default function ArxivRankedList({
  papers,
  onPaperClick,
  onClose,
}: ArxivRankedListProps) {
  const [position, setPosition] = useState({ left: 16, top: null as number | null });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Calculate initial top position (bottom: 16px)
    if (containerRef.current && position.top === null) {
      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const initialTop = windowHeight - rect.height - 16;
      setPosition({ left: 16, top: initialTop });
    }
  }, [papers.length]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const newLeft = e.clientX - dragOffset.x;
      const newTop = e.clientY - dragOffset.y;

      // Constrain to viewport bounds
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const containerWidth = containerRef.current?.offsetWidth || 450;
      const containerHeight = containerRef.current?.offsetHeight || 200;

      const constrainedLeft = Math.max(0, Math.min(newLeft, windowWidth - containerWidth));
      const constrainedTop = Math.max(0, Math.min(newTop, windowHeight - containerHeight));

      setPosition({
        left: constrainedLeft,
        top: constrainedTop,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  const handleHeaderMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setIsDragging(true);
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const getTagColor = (tag: string) => {
    if (tag.includes('HIGH_MATCH') || tag.includes('PREFERRED') || tag.includes('CODE_AVAILABLE') || tag.includes('SEMANTIC_HIGH')) {
      return '#48bb78'; // green
    }
    if (tag.includes('PENALTY') || tag.includes('NO_CODE') || tag.includes('OLDER')) {
      return '#f56565'; // red
    }
    if (tag.includes('CONTRASTIVE')) {
      return '#ed8936'; // orange
    }
    return '#4299e1'; // blue
  };

  if (papers.length === 0) {
    return null;
  }

  return (
    <div 
      ref={containerRef}
      style={{
        position: 'absolute',
        left: position.top === null ? '16px' : `${position.left}px`,
        top: position.top === null ? undefined : `${position.top}px`,
        bottom: position.top === null ? '16px' : undefined,
        zIndex: isDragging ? 10 : 5,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '8px',
        width: '450px',
        maxHeight: 'calc(100vh - 200px)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        display: 'flex',
        flexDirection: 'column',
        cursor: isDragging ? 'grabbing' : 'default',
      }}
    >
      {/* Header with close button */}
      <div 
        onMouseDown={handleHeaderMouseDown}
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #e2e8f0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0,
          cursor: 'grab',
        }}
      >
        <h3 style={{
          margin: 0,
          color: '#1a202c',
          fontSize: '16px',
          fontWeight: 600,
        }}>
          Ranked Results ({papers.length})
        </h3>
        {onClose && (
          <button
            onClick={onClose}
            onMouseDown={(e) => e.stopPropagation()}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '20px',
              cursor: 'pointer',
              color: '#718096',
              lineHeight: 1,
              padding: '0',
              width: '24px',
              height: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            aria-label="Close"
          >
            ×
          </button>
        )}
      </div>

      {/* Scrollable content */}
      <div style={{
        overflowY: 'auto',
        padding: '12px',
      }}>
        {papers.map((paper) => (
          <PaperItem
            key={paper.paper_id}
            paper={paper}
            onPaperClick={onPaperClick}
            formatScore={formatScore}
            getTagColor={getTagColor}
          />
        ))}
      </div>
    </div>
  );
}

function PaperItem({
  paper,
  onPaperClick,
  formatScore,
  getTagColor,
}: {
  paper: ScoredPaper;
  onPaperClick: (paperId: string) => void;
  formatScore: (score: number) => string;
  getTagColor: (tag: string) => string;
}) {
  const [showBreakdown, setShowBreakdown] = useState(false);

  return (
    <div
      style={{
        backgroundColor: '#f7fafc',
        borderRadius: '6px',
        padding: '12px',
        marginBottom: '12px',
        border: '1px solid #e2e8f0',
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
      <h4
        onClick={() => onPaperClick(paper.paper_id)}
        style={{
          margin: '0 0 6px 0',
          color: '#1a202c',
          fontSize: '14px',
          fontWeight: 600,
          lineHeight: 1.4,
          cursor: 'pointer',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = '#4299e1';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = '#1a202c';
        }}
      >
        {paper.title}
      </h4>

      {/* Authors */}
      <div style={{
        color: '#718096',
        fontSize: '12px',
        marginBottom: '6px',
      }}>
        {paper.authors.join(', ')}
      </div>

      {/* Published Date */}
      {paper.published && (
        <div style={{
          color: '#718096',
          fontSize: '11px',
          marginBottom: '8px',
        }}>
          Published: {paper.published}
        </div>
      )}

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' }}>
          {paper.tags.map((tag, idx) => (
            <span
              key={idx}
              style={{
                backgroundColor: getTagColor(tag),
                color: '#fff',
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: 500,
              }}
            >
              {tag.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      {/* Score Breakdown */}
      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowBreakdown(!showBreakdown);
          }}
          style={{
            background: 'none',
            border: '1px solid #cbd5e0',
            color: '#4a5568',
            padding: '6px 12px',
            borderRadius: '4px',
            fontSize: '11px',
            cursor: 'pointer',
          }}
        >
          {showBreakdown ? 'Hide' : 'Show'} Score Breakdown
        </button>
        {showBreakdown && (
          <div style={{
            marginTop: '8px',
            padding: '10px',
            backgroundColor: '#edf2f7',
            borderRadius: '4px',
            fontSize: '11px',
          }}>
            <div style={{ color: '#2d3748', marginBottom: '6px', fontWeight: 500 }}>Breakdown:</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', color: '#4a5568' }}>
              <div>Semantic: {formatScore(paper.score.breakdown.semantic_relevance)}%</div>
              <div>Keywords: {formatScore(paper.score.breakdown.must_keywords)}%</div>
              <div>Author Trust: {formatScore(paper.score.breakdown.author_trust)}%</div>
              <div>Institution: {formatScore(paper.score.breakdown.institution_trust)}%</div>
              <div>Recency: {formatScore(paper.score.breakdown.recency)}%</div>
              <div>Practicality: {formatScore(paper.score.breakdown.practicality)}%</div>
            </div>
            {paper.score.soft_penalty < 0 && (
              <div style={{ color: '#f56565', marginTop: '6px', fontSize: '10px' }}>
                Penalty: {formatScore(paper.score.soft_penalty)}%
                {paper.score.penalty_keywords.length > 0 && (
                  <span> ({paper.score.penalty_keywords.join(', ')})</span>
                )}
              </div>
            )}
            <div style={{ color: '#718096', marginTop: '6px', fontSize: '10px' }}>
              Method: {paper.score.evaluation_method}
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <a
          href={`https://arxiv.org/pdf/${paper.paper_id}.pdf`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          style={{
            padding: '6px 12px',
            borderRadius: '4px',
            border: 'none',
            backgroundColor: '#4299e1',
            color: '#fff',
            fontSize: '11px',
            fontWeight: 500,
            textDecoration: 'none',
            display: 'inline-block',
          }}
        >
          Download PDF
        </a>
        <a
          href={`https://arxiv.org/abs/${paper.paper_id}`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          style={{
            padding: '6px 12px',
            borderRadius: '4px',
            border: '1px solid #cbd5e0',
            backgroundColor: 'transparent',
            color: '#4a5568',
            fontSize: '11px',
            textDecoration: 'none',
            display: 'inline-block',
          }}
        >
          View on arXiv
        </a>
      </div>
    </div>
  );
}
