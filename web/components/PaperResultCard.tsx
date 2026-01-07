/**
 * PaperResultCard Component
 *
 * Displays a ranked paper result with score, tags, and metadata.
 */

import React, { useState } from 'react';

export interface ScoredPaper {
  rank: number;
  paper_id: string;
  title: string;
  authors: string[];
  published?: string;
  score: {
    final: number;
    breakdown: {
      semantic_relevance: number;
      must_keywords: number;
      author_trust: number;
      institution_trust: number;
      recency: number;
      practicality: number;
    };
    soft_penalty: number;
    penalty_keywords: string[];
    evaluation_method: string;
  };
  tags: string[];
  local_status: {
    already_downloaded: boolean;
    local_path: string | null;
  };
  original_data: any;
}

interface PaperResultCardProps {
  paper: ScoredPaper;
  onDownloadPdf?: (paperId: string) => void;
}

export default function PaperResultCard({ paper, onDownloadPdf }: PaperResultCardProps) {
  const [showBreakdown, setShowBreakdown] = useState(false);

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const getTagColor = (tag: string) => {
    if (tag.includes('HIGH_MATCH') || tag.includes('PREFERRED') || tag.includes('CODE_AVAILABLE')) {
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

  return (
    <div style={{
      backgroundColor: '#1a202c',
      borderRadius: '8px',
      padding: '16px',
      marginBottom: '12px',
      border: '1px solid #2d3748',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <span style={{
              backgroundColor: '#4a90d9',
              color: '#fff',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 600,
            }}>
              #{paper.rank}
            </span>
            <span style={{
              backgroundColor: '#2d3748',
              color: '#e2e8f0',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 500,
            }}>
              Score: {formatScore(paper.score.final)}%
            </span>
          </div>
          <h3 style={{
            margin: 0,
            color: '#fff',
            fontSize: '16px',
            fontWeight: 600,
            lineHeight: 1.4,
            marginBottom: '8px',
          }}>
            {paper.title}
          </h3>
          <div style={{ color: '#a0aec0', fontSize: '13px', marginBottom: '4px' }}>
            {paper.authors.join(', ')}
          </div>
          {paper.published && (
            <div style={{ color: '#718096', fontSize: '12px' }}>
              Published: {paper.published}
            </div>
          )}
        </div>
      </div>

      {/* Tags */}
      {paper.tags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '12px' }}>
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
      <div style={{ marginBottom: '12px' }}>
        <button
          onClick={() => setShowBreakdown(!showBreakdown)}
          style={{
            background: 'none',
            border: '1px solid #4a5568',
            color: '#a0aec0',
            padding: '6px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          {showBreakdown ? 'Hide' : 'Show'} Score Breakdown
        </button>
        {showBreakdown && (
          <div style={{
            marginTop: '8px',
            padding: '12px',
            backgroundColor: '#2d3748',
            borderRadius: '4px',
            fontSize: '12px',
          }}>
            <div style={{ color: '#e2e8f0', marginBottom: '8px' }}>Breakdown:</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', color: '#a0aec0' }}>
              <div>Semantic: {formatScore(paper.score.breakdown.semantic_relevance)}%</div>
              <div>Keywords: {formatScore(paper.score.breakdown.must_keywords)}%</div>
              <div>Author Trust: {formatScore(paper.score.breakdown.author_trust)}%</div>
              <div>Institution: {formatScore(paper.score.breakdown.institution_trust)}%</div>
              <div>Recency: {formatScore(paper.score.breakdown.recency)}%</div>
              <div>Practicality: {formatScore(paper.score.breakdown.practicality)}%</div>
            </div>
            {paper.score.soft_penalty < 0 && (
              <div style={{ color: '#f56565', marginTop: '8px' }}>
                Penalty: {formatScore(paper.score.soft_penalty)}%
                {paper.score.penalty_keywords.length > 0 && (
                  <span> ({paper.score.penalty_keywords.join(', ')})</span>
                )}
              </div>
            )}
            <div style={{ color: '#718096', marginTop: '8px', fontSize: '11px' }}>
              Method: {paper.score.evaluation_method}
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '8px' }}>
        {paper.local_status.already_downloaded && (
          <span style={{
            color: '#48bb78',
            fontSize: '12px',
            padding: '4px 8px',
            backgroundColor: '#2d3748',
            borderRadius: '4px',
          }}>
            ✓ Downloaded
          </span>
        )}
        {onDownloadPdf && (
          <button
            onClick={() => onDownloadPdf(paper.paper_id)}
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              border: 'none',
              backgroundColor: '#4a90d9',
              color: '#fff',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Download PDF
          </button>
        )}
        <a
          href={`https://arxiv.org/abs/${paper.paper_id}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            padding: '6px 12px',
            borderRadius: '4px',
            border: '1px solid #4a5568',
            backgroundColor: 'transparent',
            color: '#a0aec0',
            fontSize: '12px',
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

