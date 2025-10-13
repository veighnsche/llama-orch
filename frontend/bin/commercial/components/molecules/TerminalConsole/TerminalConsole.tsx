import { ReactNode } from 'react'

interface TerminalConsoleProps {
  title: string
  children: ReactNode
  footer?: ReactNode
  ariaLabel?: string
}

export function TerminalConsole({ title, children, footer, ariaLabel }: TerminalConsoleProps) {
  return (
    <div className="rounded-2xl border border-border bg-card overflow-hidden" role="region" aria-label={ariaLabel}>
      {/* Terminal top bar */}
      <div className="flex items-center gap-1 bg-muted/50 px-4 py-2">
        <span className="size-2 rounded-full bg-red-500/70" aria-hidden="true" />
        <span className="size-2 rounded-full bg-yellow-500/70" aria-hidden="true" />
        <span className="size-2 rounded-full bg-green-500/70" aria-hidden="true" />
        <span className="ml-3 font-mono text-xs text-muted-foreground">{title}</span>
      </div>

      {/* Console content */}
      <div className="bg-background p-6 font-mono text-sm leading-relaxed">{children}</div>

      {/* Optional footer */}
      {footer && <div className="border-t border-border px-6 py-3">{footer}</div>}
    </div>
  )
}
