// Created by: TEAM-AI-ASSISTANT
import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface ConsoleOutputProps {
  /** Console content - can be string or ReactNode for syntax highlighting */
  children: ReactNode
  /** Show terminal window chrome (traffic lights, title bar) */
  showChrome?: boolean
  /** Terminal title */
  title?: string
  /** Terminal variant */
  variant?: 'terminal' | 'code' | 'output'
  /** Additional CSS classes */
  className?: string
  /** Background style */
  background?: 'dark' | 'light' | 'card'
}

/**
 * ConsoleOutput - A component for displaying terminal/console output with proper monospace font
 * 
 * Features:
 * - Uses Geist Mono font for authentic console appearance
 * - Optional terminal window chrome (macOS-style traffic lights)
 * - Multiple variants for different use cases
 * - Dark/light background options
 * - Proper text selection and overflow handling
 * 
 * @example
 * ```tsx
 * <ConsoleOutput showChrome title="bash">
 *   $ npm install rbee
 * </ConsoleOutput>
 * ```
 */
export function ConsoleOutput({
  children,
  showChrome = false,
  title,
  variant = 'terminal',
  className,
  background = 'dark',
}: ConsoleOutputProps) {
  const bgStyles = {
    dark: 'bg-slate-950 text-slate-50',
    light: 'bg-background text-foreground',
    card: 'bg-card text-card-foreground',
  }

  return (
    <div
      className={cn(
        'overflow-hidden rounded-lg border border-border shadow-sm',
        className
      )}
    >
      {showChrome && (
        <div className="flex items-center gap-2 border-b border-border bg-muted px-4 py-3">
          <div className="flex gap-2">
            <div className="h-3 w-3 rounded-full bg-terminal-red" />
            <div className="h-3 w-3 rounded-full bg-terminal-amber" />
            <div className="h-3 w-3 rounded-full bg-terminal-green" />
          </div>
          {title && (
            <span className="ml-2 font-mono text-sm text-muted-foreground">
              {title}
            </span>
          )}
        </div>
      )}
      <div
        className={cn(
          'overflow-x-auto p-4 text-sm leading-relaxed font-mono',
          bgStyles[background]
        )}
      >
        {children}
      </div>
    </div>
  )
}
