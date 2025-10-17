import type { ReactNode } from 'react'

/**
 * Lightweight inline markdown parser for UI strings
 * 
 * Supports:
 * - **bold** → <strong>
 * - *italic* → <em>
 * - [link text](url) → <a>
 * 
 * Use for: Card descriptions, tooltips, short UI copy
 * Don't use for: Long-form content, complex markdown
 * 
 * @example
 * parseInlineMarkdown('Power **your** GPUs')
 * // Returns: ['Power ', <strong key="0">your</strong>, ' GPUs']
 */
export function parseInlineMarkdown(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  let currentIndex = 0
  let keyCounter = 0

  // Combined regex that matches bold, italic, or links
  // IMPORTANT: Bold (**) must come before italic (*) in the alternation
  const combinedRegex = /(\*\*([^*]+?)\*\*)|(\*([^*]+?)\*)|(\[([^\]]+?)\]\(([^)]+?)\))/g
  
  let match: RegExpExecArray | null

  while ((match = combinedRegex.exec(text)) !== null) {
    // Add text before this match
    if (match.index > currentIndex) {
      parts.push(text.slice(currentIndex, match.index))
    }

    // Determine which pattern matched
    if (match[1]) {
      // Bold: **text**
      parts.push(<strong key={keyCounter++}>{match[2]}</strong>)
    } else if (match[3]) {
      // Italic: *text*
      parts.push(<em key={keyCounter++}>{match[4]}</em>)
    } else if (match[5]) {
      // Link: [text](url)
      const linkText = match[6]
      const url = match[7]
      parts.push(
        <a
          key={keyCounter++}
          href={url}
          className="text-[color:var(--primary)] underline underline-offset-2 decoration-amber-300 hover:text-[color:var(--accent)] hover:decoration-amber-400"
          target={url?.startsWith('http') ? '_blank' : undefined}
          rel={url?.startsWith('http') ? 'noopener noreferrer' : undefined}
        >
          {linkText}
        </a>
      )
    }

    currentIndex = match.index + match[0].length
  }

  // Add remaining text
  if (currentIndex < text.length) {
    parts.push(text.slice(currentIndex))
  }

  return parts.length > 0 ? parts : [text]
}

/**
 * React component wrapper for inline markdown
 * 
 * @example
 * <InlineMarkdown>Power **your** GPUs with *zero* API fees</InlineMarkdown>
 */
export function InlineMarkdown({ children }: { children: string }) {
  return <>{parseInlineMarkdown(children)}</>
}
