import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface TerminalWindowProps {
  /** Terminal title */
  title?: string
  /** Terminal content */
  children: ReactNode
  /** Terminal variant */
  variant?: 'terminal' | 'code' | 'output'
  /** Additional CSS classes */
  className?: string
}

export function TerminalWindow({
  title,
  children,
  variant = 'terminal',
  className,
}: TerminalWindowProps) {
  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg overflow-hidden shadow-2xl',
        className
      )}
    >
      <div className="flex items-center gap-2 px-4 py-3 bg-muted border-b border-border">
        <div className="flex gap-2">
          <div className="h-3 w-3 rounded-full bg-terminal-red"></div>
          <div className="h-3 w-3 rounded-full bg-terminal-amber"></div>
          <div className="h-3 w-3 rounded-full bg-terminal-green"></div>
        </div>
        {title && (
          <span className="text-muted-foreground text-sm ml-2 font-mono">
            {title}
          </span>
        )}
      </div>
      <div className="p-6 font-mono text-sm">{children}</div>
    </div>
  )
}
