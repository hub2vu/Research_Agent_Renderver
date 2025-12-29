/**
 * PaperCard Component
 *
 * Displays paper metadata in a card format.
 */

import React from 'react';
import { GraphNode } from '../api/mcp';

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
          {node.title || node.id}
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
        {node.title || node.id}
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
          <p
            style={{
              margin: 0,
              fontSize: '12px',
              color: '#4a5568',
              lineHeight: 1.6,
              maxHeight: '200px',
              overflow: 'auto'
            }}
          >
            {node.abstract}
          </p>
        </div>
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
