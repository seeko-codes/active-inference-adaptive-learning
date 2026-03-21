import 'katex/dist/katex.min.css';
import katex from 'katex';
import { useMemo } from 'react';

function algebraToLatex(expr: string): string {
  let s = expr;
  // Convert * to \cdot for display
  s = s.replace(/\*/g, ' \\cdot ');
  // Wrap variables in math italic (single lowercase letters)
  s = s.replace(/\b([a-z])\b/g, '{$1}');
  return s;
}

export function MathDisplay({ expr, display = false }: { expr: string; display?: boolean }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(algebraToLatex(expr), {
        displayMode: display,
        throwOnError: false,
      });
    } catch {
      return expr;
    }
  }, [expr, display]);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
