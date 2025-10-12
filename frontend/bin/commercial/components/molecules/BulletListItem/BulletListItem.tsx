import { cn } from '@/lib/utils'

export interface BulletListItemProps {
  /** Item title */
  title: string
  /** Optional description */
  description?: string
  /** Bullet color (Tailwind class) */
  color?: string
  /** Bullet variant */
  variant?: 'dot' | 'check' | 'arrow'
  /** Additional CSS classes */
  className?: string
}

export function BulletListItem({
  title,
  description,
  color = 'chart-3',
  variant = 'dot',
  className,
}: BulletListItemProps) {
  const renderBullet = () => {
    switch (variant) {
      case 'dot':
        return (
          <div
            className={cn(
              'h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5',
              `bg-${color}/20`
            )}
          >
            <div className={cn('h-2 w-2 rounded-full', `bg-${color}`)}></div>
          </div>
        )
      case 'check':
        return (
          <div
            className={cn(
              'h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5',
              `bg-${color}/20 text-${color}`
            )}
          >
            ✓
          </div>
        )
      case 'arrow':
        return (
          <div
            className={cn(
              'h-6 w-6 flex items-center justify-center flex-shrink-0 mt-0.5',
              `text-${color}`
            )}
          >
            →
          </div>
        )
    }
  }

  return (
    <li className={cn('flex items-start gap-3', className)}>
      {renderBullet()}
      <div>
        <div className="font-medium text-foreground">{title}</div>
        {description && (
          <div className="text-sm text-muted-foreground">{description}</div>
        )}
      </div>
    </li>
  )
}
