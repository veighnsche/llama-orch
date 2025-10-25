import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface TabButtonProps {
  id: string
  label: string
  icon: ReactNode
  active: boolean
  onClick: () => void
  className?: string
}

export function TabButton({ label, icon: Icon, active, onClick, className }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex items-center gap-2 rounded border px-4 py-2 text-sm font-medium transition-all',
        active
          ? 'border-primary bg-primary/10 text-primary'
          : 'border-border bg-card text-muted-foreground hover:border-border hover:text-foreground',
        className,
      )}
    >
      <div className="h-4 w-4">{Icon}</div>
      {label}
    </button>
  )
}
