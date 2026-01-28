/**
 * LatexText Component
 *
 * Renders plain text with LaTeX math expressions.
 * Supports both inline ($...$) and display ($$...$$) math.
 * Uses KaTeX for rendering.
 */

import React, { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface LatexTextProps {
  children: string;
  style?: React.CSSProperties;
  className?: string;
}

interface TextPart {
  type: 'text' | 'inline-math' | 'display-math';
  content: string;
}

/**
 * Parse text and extract LaTeX math expressions
 */
function parseLatex(text: string): TextPart[] {
  if (!text) return [];

  const parts: TextPart[] = [];
  let remaining = text;

  // Regex patterns for LaTeX
  // Display math: $$...$$ (greedy, multiline)
  // Inline math: $...$ (non-greedy, single line)
  const displayMathRegex = /\$\$([\s\S]+?)\$\$/;
  const inlineMathRegex = /\$([^\$\n]+?)\$/;

  while (remaining.length > 0) {
    const displayMatch = remaining.match(displayMathRegex);
    const inlineMatch = remaining.match(inlineMathRegex);

    // Find the earliest match
    let earliestMatch: { match: RegExpMatchArray; type: 'inline-math' | 'display-math' } | null = null;

    if (displayMatch && displayMatch.index !== undefined) {
      earliestMatch = { match: displayMatch, type: 'display-math' };
    }

    if (inlineMatch && inlineMatch.index !== undefined) {
      if (!earliestMatch || inlineMatch.index < earliestMatch.match.index!) {
        earliestMatch = { match: inlineMatch, type: 'inline-math' };
      }
    }

    if (!earliestMatch) {
      // No more math expressions, add remaining text
      if (remaining) {
        parts.push({ type: 'text', content: remaining });
      }
      break;
    }

    const { match, type } = earliestMatch;
    const matchIndex = match.index!;

    // Add text before the match
    if (matchIndex > 0) {
      parts.push({ type: 'text', content: remaining.slice(0, matchIndex) });
    }

    // Add the math expression
    parts.push({ type, content: match[1] });

    // Continue with the rest
    remaining = remaining.slice(matchIndex + match[0].length);
  }

  return parts;
}

/**
 * Render a single math expression using KaTeX
 */
function renderMath(latex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      errorColor: '#cc0000',
      strict: false,
      trust: true,
    });
  } catch (e) {
    console.error('KaTeX rendering error:', e);
    return `<span style="color: #cc0000;">${latex}</span>`;
  }
}

export default function LatexText({ children, style, className }: LatexTextProps) {
  const parts = useMemo(() => parseLatex(children || ''), [children]);

  if (parts.length === 0) {
    return null;
  }

  // If no math expressions found, return plain text
  if (parts.length === 1 && parts[0].type === 'text') {
    return (
      <span style={style} className={className}>
        {parts[0].content}
      </span>
    );
  }

  return (
    <span style={style} className={className}>
      {parts.map((part, index) => {
        if (part.type === 'text') {
          return <span key={index}>{part.content}</span>;
        }

        const isDisplay = part.type === 'display-math';
        const html = renderMath(part.content, isDisplay);

        if (isDisplay) {
          return (
            <span
              key={index}
              style={{ display: 'block', textAlign: 'center', margin: '0.5em 0' }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        }

        return (
          <span
            key={index}
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </span>
  );
}

/**
 * LatexDiv - Block-level component for rendering text with LaTeX
 * Preserves whitespace and line breaks (like pre-wrap)
 */
export function LatexDiv({ children, style, className }: LatexTextProps) {
  const lines = useMemo(() => (children || '').split('\n'), [children]);

  return (
    <div style={style} className={className}>
      {lines.map((line, lineIndex) => (
        <React.Fragment key={lineIndex}>
          <LatexText>{line}</LatexText>
          {lineIndex < lines.length - 1 && <br />}
        </React.Fragment>
      ))}
    </div>
  );
}
