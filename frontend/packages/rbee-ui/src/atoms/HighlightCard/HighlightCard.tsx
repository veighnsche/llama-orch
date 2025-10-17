import { BulletListItem } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { cva, type VariantProps } from 'class-variance-authority'
import { Card, CardContent, CardHeader, CardTitle } from '../Card/Card'

const highlightCardVariants = cva('rounded-lg border p-4', {
  variants: {
    color: {
      primary: 'border-primary/50 bg-primary/10',
      'chart-1': 'border-chart-1/50 bg-chart-1/10',
      'chart-2': 'border-chart-2/50 bg-chart-2/10',
      'chart-3': 'border-chart-3/50 bg-chart-3/10',
      'chart-4': 'border-chart-4/50 bg-chart-4/10',
      'chart-5': 'border-chart-5/50 bg-chart-5/10',
    },
  },
  defaultVariants: {
    color: 'chart-3',
  },
})

const highlightCardTitleVariants = cva('mb-2 font-semibold', {
  variants: {
    color: {
      primary: 'text-primary',
      'chart-1': 'text-chart-1',
      'chart-2': 'text-chart-2',
      'chart-3': 'text-chart-3',
      'chart-4': 'text-chart-4',
      'chart-5': 'text-chart-5',
    },
  },
  defaultVariants: {
    color: 'chart-3',
  },
})

export interface HighlightCardProps extends VariantProps<typeof highlightCardVariants> {
  /** Card heading */
  heading: string
  /** List of items to display */
  items: string[]
  /** Color for checkmark icons (overrides default color) */
  checkmarkColor?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5' | 'white'
  /** Show checkmarks as disabled/muted */
  disabledCheckmarks?: boolean
  /** Additional CSS classes */
  className?: string
}

/**
 * HighlightCard - A highlighted card with a title and bullet list
 *
 * Uses Card atom with CardHeader/CardTitle and BulletListItem for consistent styling.
 *
 * @example
 * ```tsx
 * <HighlightCard
 *   heading="Key Features"
 *   items={['Feature 1', 'Feature 2', 'Feature 3']}
 *   color="chart-3"
 * />
 * ```
 */
export function HighlightCard({
  heading,
  items,
  color = 'chart-3',
  checkmarkColor,
  disabledCheckmarks = false,
  className,
}: HighlightCardProps) {
  // Determine the effective checkmark color
  // If disabled, use 'muted', otherwise use checkmarkColor or default color
  const effectiveCheckmarkColor = disabledCheckmarks ? 'muted' : (checkmarkColor || color)

  return (
    <Card className={cn(highlightCardVariants({ color }), 'shadow-none', className)}>
      <CardHeader className="pb-0 p-0 mb-2">
        <CardTitle className={highlightCardTitleVariants({ color })}>{heading}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ul className="space-y-1">
          {items.map((item, idx) => (
            <BulletListItem
              key={idx}
              title={item}
              variant="check"
              showPlate={false}
              color={effectiveCheckmarkColor as any}
              className={cn(
                'text-xs',
                disabledCheckmarks ? 'text-muted-foreground/50' : 'text-foreground/85',
              )}
            />
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
