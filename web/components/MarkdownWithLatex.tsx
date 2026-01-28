/**
 * MarkdownWithLatex Component
 *
 * ReactMarkdown wrapper with LaTeX math support via remark-math and rehype-katex.
 * Supports both inline ($...$) and display ($$...$$) math in markdown.
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownWithLatexProps {
  children: string;
  components?: Record<string, React.ComponentType<any>>;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Default markdown component styles (academic paper style)
 */
export const defaultMarkdownComponents = {
  // Table styles
  table: ({ node, ...props }: any) => (
    <table
      style={{
        borderCollapse: 'collapse',
        width: '100%',
        margin: '20px 0',
        fontSize: '13px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      }}
      {...props}
    />
  ),
  thead: ({ node, ...props }: any) => (
    <thead
      style={{ backgroundColor: '#f7fafc', borderBottom: '2px solid #e2e8f0' }}
      {...props}
    />
  ),
  th: ({ node, ...props }: any) => (
    <th
      style={{
        padding: '12px',
        textAlign: 'left',
        fontWeight: 700,
        color: '#2d3748',
        border: '1px solid #e2e8f0',
      }}
      {...props}
    />
  ),
  td: ({ node, ...props }: any) => (
    <td
      style={{
        padding: '12px',
        border: '1px solid #e2e8f0',
        color: '#4a5568',
        verticalAlign: 'top',
      }}
      {...props}
    />
  ),
  // Header styles
  h1: ({ node, ...props }: any) => (
    <h1
      style={{
        fontSize: '20px',
        fontWeight: 800,
        color: '#2b6cb0',
        marginTop: '24px',
        marginBottom: '16px',
        borderBottom: '1px solid #bee3f8',
        paddingBottom: '8px',
      }}
      {...props}
    />
  ),
  h2: ({ node, ...props }: any) => (
    <h2
      style={{
        fontSize: '16px',
        fontWeight: 700,
        color: '#2c5282',
        marginTop: '20px',
        marginBottom: '12px',
        borderLeft: '4px solid #4299e1',
        paddingLeft: '10px',
      }}
      {...props}
    />
  ),
  h3: ({ node, ...props }: any) => (
    <h3
      style={{
        fontSize: '14px',
        fontWeight: 700,
        color: '#2d3748',
        marginTop: '16px',
        marginBottom: '8px',
      }}
      {...props}
    />
  ),
  // Body and list styles
  p: ({ node, ...props }: any) => (
    <p
      style={{
        lineHeight: 1.7,
        marginBottom: '12px',
        fontSize: '13.5px',
        color: '#1a202c',
      }}
      {...props}
    />
  ),
  ul: ({ node, ...props }: any) => (
    <ul style={{ paddingLeft: '20px', marginBottom: '16px' }} {...props} />
  ),
  li: ({ node, ...props }: any) => (
    <li style={{ marginBottom: '6px', lineHeight: 1.6 }} {...props} />
  ),
  strong: ({ node, ...props }: any) => (
    <strong style={{ color: '#2b6cb0', fontWeight: 600 }} {...props} />
  ),
  // Code block styles
  code: ({ node, inline, className, children, ...props }: any) => {
    if (inline) {
      return (
        <code
          style={{
            backgroundColor: '#edf2f7',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.9em',
            fontFamily: 'monospace',
          }}
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        style={{
          display: 'block',
          backgroundColor: '#1a202c',
          color: '#e2e8f0',
          padding: '16px',
          borderRadius: '8px',
          overflowX: 'auto',
          fontSize: '13px',
          fontFamily: 'monospace',
        }}
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({ node, ...props }: any) => (
    <pre style={{ margin: '16px 0' }} {...props} />
  ),
};

export default function MarkdownWithLatex({
  children,
  components,
  className,
  style,
}: MarkdownWithLatexProps) {
  const mergedComponents = {
    ...defaultMarkdownComponents,
    ...components,
  };

  return (
    <div className={className} style={style}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={mergedComponents}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
