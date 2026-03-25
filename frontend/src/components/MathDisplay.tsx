import 'katex/dist/katex.min.css';
import katex from 'katex';
import { useMemo } from 'react';

function algebraToLatex(expr: string): string {
  let s = expr;
  s = s.replace(/\*/g, ' \\cdot ');
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

  return (
    <span
      className="leading-none"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
