import { StepNumber } from '@rbee/ui/molecules/StepNumber'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface StepListItemProps {
  /** Step number */
  number: number
  /** Step title */
  title: string
  /** Step description/body */
  body: string
  /** Optional custom icon/element instead of number */
  icon?: ReactNode
  /** Additional CSS classes */
  className?: string
}

/**
 * StepListItem - A molecule for displaying numbered steps in a list
 *
 * Features:
 * - Numbered badge using StepNumber component
 * - Two-line layout with title and description
 * - Optional custom icon support
 * - Grid-based layout for alignment
 *
 * @example
 * ```tsx
 * <ol className="space-y-6">
 *   <StepListItem
 *     number={1}
 *     title="Install rbee"
 *     body="Run one command on Windows, macOS, or Linux."
 *   />
 *   <StepListItem
 *     number={2}
 *     title="Add Your Hardware"
 *     body="rbee auto-detects GPUs and CPUs across your network."
 *   />
 * </ol>
 * ```
 */
export function StepListItem({ number, title, body, icon, className }: StepListItemProps) {
  return (
    <li
      className={cn('grid grid-cols-[auto_1fr] items-start gap-4', className)}
      aria-label={`Step ${number}: ${title}`}
    >
      {icon || <StepNumber number={number} size="sm" variant="outline" className="bg-primary/10 border-0" />}
      <div>
        <div className="mb-1 font-semibold text-foreground">{title}</div>
        <div className="text-sm leading-relaxed text-muted-foreground">{body}</div>
      </div>
    </li>
  )
}
