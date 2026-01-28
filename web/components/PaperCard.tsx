/**
 * PaperCard Component
 *
 * Displays paper metadata in a card format.
 */

import React from 'react';
import { GraphNode } from '../lib/mcp';
import LatexText, { LatexDiv } from './LatexText';

interface PaperCardProps {
  node: GraphNode;
  compact?: boolean;
}

export default function PaperCard({ node, compact = false }: PaperCardProps) {
  if (compact) {
    return (
      <div
        style={{
          padding: '12px',
          backgroundColor: '#f7fafc',
          borderRadius: '6px',
          borderLeft: '3px solid #4299e1'
        }}
      >
        <h4 style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#2d3748' }}>
          <LatexText>{node.title || node.id}</LatexText>
        </h4>
        {node.authors && node.authors.length > 0 && (
          <p style={{ margin: 0, fontSize: '11px', color: '#718096' }}>
            {node.authors.slice(0, 2).join(', ')}
            {node.authors.length > 2 && ' et al.'}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      {/* Title */}
      <h4
        style={{
          margin: '0 0 12px 0',
          fontSize: '16px',
          color: '#2d3748',
          lineHeight: 1.4
        }}
      >
        <LatexText>{node.title || node.id}</LatexText>
      </h4>

      {/* Paper ID */}
      <div
        style={{
          display: 'inline-block',
          padding: '2px 8px',
          backgroundColor: '#edf2f7',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#718096',
          marginBottom: '12px'
        }}
      >
        {node.id}
      </div>

      {/* Authors */}
      {node.authors && node.authors.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              color: '#a0aec0',
              marginBottom: '4px',
              textTransform: 'uppercase'
            }}
          >
            Authors
          </label>
          <p style={{ margin: 0, fontSize: '13px', color: '#4a5568' }}>
            {node.authors.join(', ')}
          </p>
        </div>
      )}

      {/* Year */}
      {node.year && (
        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              color: '#a0aec0',
              marginBottom: '4px',
              textTransform: 'uppercase'
            }}
          >
            Year
          </label>
          <p style={{ margin: 0, fontSize: '13px', color: '#4a5568' }}>
            {node.year}
          </p>
        </div>
      )}

      {/* Cluster (for Graph B) */}
      {node.cluster !== undefined && (
        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              color: '#a0aec0',
              marginBottom: '4px',
              textTransform: 'uppercase'
            }}
          >
            Cluster
          </label>
          <span
            style={{
              display: 'inline-block',
              padding: '2px 8px',
              backgroundColor: getClusterColor(node.cluster),
              color: '#fff',
              borderRadius: '4px',
              fontSize: '12px'
            }}
          >
            Cluster {node.cluster}
          </span>
        </div>
      )}

      {/* Depth (for Graph A) */}
      {node.depth !== undefined && (
        <div style={{ marginBottom: '12px' }}>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              color: '#a0aec0',
              marginBottom: '4px',
              textTransform: 'uppercase'
            }}
          >
            Reference Depth
          </label>
          <p style={{ margin: 0, fontSize: '13px', color: '#4a5568' }}>
            {node.depth === 0 ? 'Center' : `Level ${node.depth}`}
          </p>
        </div>
      )}

      {/* Abstract */}
      {node.abstract && (
        <div>
          <label
            style={{
              display: 'block',
              fontSize: '11px',
              color: '#a0aec0',
              marginBottom: '4px',
              textTransform: 'uppercase'
            }}
          >
            Abstract
          </label>
          <div
            style={{
              margin: 0,
              fontSize: '12px',
              color: '#4a5568',
              lineHeight: 1.6,
              maxHeight: '200px',
              overflow: 'auto'
            }}
          >
            <LatexDiv>{node.abstract}</LatexDiv>
          </div>
        </div>
      )}

      {/* Search Result Metadata (reasoning, tags, pdf_url) */}
      {(node as any).scoredPaper && (
        <>
          {/* Reasoning */}
          {(node as any).scoredPaper.reasoning && (
            <div style={{ marginTop: '12px' }}>
              <label
                style={{
                  display: 'block',
                  fontSize: '11px',
                  color: '#a0aec0',
                  marginBottom: '4px',
                  textTransform: 'uppercase'
                }}
              >
                💡 Why Selected
              </label>
              <div
                style={{
                  margin: 0,
                  fontSize: '12px',
                  color: '#4a5568',
                  fontStyle: 'italic',
                  padding: '8px',
                  backgroundColor: '#edf2f7',
                  borderRadius: '4px',
                }}
              >
                <LatexDiv>{(node as any).scoredPaper.reasoning}</LatexDiv>
              </div>
            </div>
          )}

          {/* Tags */}
          {(node as any).scoredPaper.tags && (node as any).scoredPaper.tags.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <label
                style={{
                  display: 'block',
                  fontSize: '11px',
                  color: '#a0aec0',
                  marginBottom: '4px',
                  textTransform: 'uppercase'
                }}
              >
                Tags
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {(node as any).scoredPaper.tags.map((tag: string, idx: number) => {
                  const getTagColor = (tag: string) => {
                    if (tag.includes('HIGH_MATCH') || tag.includes('PREFERRED') || tag.includes('CODE_AVAILABLE')) {
                      return '#48bb78';
                    }
                    if (tag.includes('PENALTY') || tag.includes('NO_CODE') || tag.includes('OLDER')) {
                      return '#f56565';
                    }
                    return '#4299e1';
                  };
                  return (
                    <span
                      key={idx}
                      style={{
                        backgroundColor: getTagColor(tag),
                        color: '#fff',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '11px',
                      }}
                    >
                      {tag.replace(/_/g, ' ')}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* PDF Download Link */}
          {(node as any).scoredPaper.original_data?.pdf_url && (
            <div style={{ marginTop: '12px' }}>
              <a
                href={(node as any).scoredPaper.original_data.pdf_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'inline-block',
                  padding: '8px 16px',
                  backgroundColor: '#4299e1',
                  color: '#fff',
                  borderRadius: '4px',
                  textDecoration: 'none',
                  fontSize: '13px',
                  fontWeight: 500,
                }}
              >
                📄 Download PDF
              </a>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function getClusterColor(cluster: number): string {
  const colors = [
    '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
    '#f56565', '#38b2ac', '#ed64a6', '#667eea'
  ];
  return colors[cluster % colors.length];
}
