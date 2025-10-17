import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utility function to merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Re-export focus ring utilities
export { brandLink, focusRing, focusRingDestructive, focusRingTight } from './focus-ring'

// Re-export inline markdown parser
export { InlineMarkdown, parseInlineMarkdown } from './parse-inline-markdown'
