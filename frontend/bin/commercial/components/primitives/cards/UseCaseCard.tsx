import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface UseCaseCardProps {
  icon: LucideIcon
  title: string
  subtitle: string
  description: string
  challenges: string[]
  solutions: string[]
  className?: string
}

export function UseCaseCard({
  icon: Icon,
  title,
  subtitle,
  description,
  challenges,
  solutions,
  className,
}: UseCaseCardProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-card p-8', className)}>
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>
      </div>

      <p className="mb-4 leading-relaxed text-muted-foreground">{description}</p>

      <div className="mb-4 rounded-lg border border-border bg-background p-4">
        <div className="mb-2 font-semibold text-foreground">Challenge:</div>
        <ul className="space-y-1 text-sm text-muted-foreground">
          {challenges.map((challenge, index) => (
            <li key={index}>• {challenge}</li>
          ))}
        </ul>
      </div>

      <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
        <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
        <ul className="space-y-1 text-sm text-muted-foreground">
          {solutions.map((solution, index) => (
            <li key={index}>• {solution}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}
