import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { BulletListItem } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'

export interface ListCardProps {
  /** Card title */
  title: string
  /** List items */
  items: string[]
  /** Bullet variant */
  variant?: 'dot' | 'check' | 'arrow'
  /** Bullet color */
  color?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5' | 'white' | 'muted'
  /** Show background plate for bullets */
  showPlate?: boolean
  /** Card background color class */
  cardClassName?: string
  /** Title color class */
  titleClassName?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * ListCard molecule - a card with a title and bullet list
 * Commonly used for challenges, solutions, features, or benefits
 *
 * @example
 * <ListCard
 *   title="Challenge"
 *   items={['Item 1', 'Item 2']}
 *   variant="dot"
 *   color="muted"
 * />
 *
 * @example
 * <ListCard
 *   title="Solution with rbee"
 *   items={['Solution 1', 'Solution 2']}
 *   variant="check"
 *   color="chart-3"
 *   cardClassName="border-chart-3/50 bg-chart-3/10"
 *   titleClassName="text-chart-3"
 * />
 */
export function ListCard({
  title,
  items,
  variant = 'dot',
  color = 'muted',
  showPlate = false,
  cardClassName,
  titleClassName,
  className,
}: ListCardProps) {
  return (
    <Card className={cn('border bg-background', cardClassName, className)}>
      <CardHeader className="pb-3">
        <CardTitle className={cn('text-base', titleClassName)}>{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0 px-6 pb-4">
        <ul className="space-y-1.5">
          {items.map((item) => (
            <BulletListItem key={item} title={item} variant={variant} color={color} showPlate={showPlate} />
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
