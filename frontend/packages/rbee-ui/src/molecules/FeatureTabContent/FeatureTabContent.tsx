import type { ReactNode } from 'react'

export interface FeatureTabContentProps {
  children: ReactNode
}

/**
 * A styled card wrapper for feature tab content with consistent spacing,
 * borders, animations, and motion-reduce support.
 */
export function FeatureTabContent({ children }: FeatureTabContentProps) {
  return (
    <div className="rounded border border-border bg-card p-6 md:p-8 space-y-6 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
      {children}
    </div>
  )
}
