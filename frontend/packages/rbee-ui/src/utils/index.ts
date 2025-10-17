import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utility function to merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Re-export focus ring utilities
export { focusRing, focusRingTight, focusRingDestructive, brandLink } from './focus-ring'

// Re-export inline markdown parser
export { parseInlineMarkdown, InlineMarkdown } from './parse-inline-markdown'
