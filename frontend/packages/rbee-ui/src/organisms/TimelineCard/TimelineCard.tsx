import { Card, CardContent, CardDescription, CardHeader, CardTitle, Progress } from '@rbee/ui/atoms'
import { TimelineStep } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'

export interface TimelineCardProps {
  /** Card heading */
  heading: string
  /** Card description */
  description: string
  /** Progress value (0-100) */
  progress?: number
  /** Timeline items */
  weeks: Array<{
    week: string
    phase: string
  }>
  /** Additional CSS classes */
  className?: string
}

/**
 * TimelineCard organism - displays a timeline with progress indicator
 * Used for showing deployment timelines, project phases, or multi-week schedules
 *
 * @example
 * <TimelineCard
 *   heading="Deployment Timeline"
 *   description="Your custom deployment schedule"
 *   progress={25}
 *   weeks={[
 *     { week: 'Week 1', phase: 'Infrastructure Setup' },
 *     { week: 'Week 2', phase: 'Model Deployment' }
 *   ]}
 * />
 */
export function TimelineCard({ heading, description, progress = 25, weeks, className }: TimelineCardProps) {
  return (
    <Card className={cn('border-primary/20 bg-primary/10', className)}>
      <CardHeader>
        <CardTitle className="text-xl">{heading}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Progress bar */}
        <Progress value={progress} className="h-1" aria-label={`${progress}% complete`} />

        {/* Week chips */}
        <ol className="space-y-3">
          {weeks.map((week, idx) => (
            <li key={idx}>
              <TimelineStep
                timestamp={week.week}
                title={week.phase}
                className="px-3 py-2 hover:bg-secondary hover:ring-0"
              />
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}
