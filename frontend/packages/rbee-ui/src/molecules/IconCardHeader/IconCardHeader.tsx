import { CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

export interface IconCardHeaderProps {
  /** Rendered icon component */
  icon: React.ReactNode
  /** Card title */
  title: string
  /** Optional subtitle/description */
  subtitle?: string
  /** ID for the title (for aria-labelledby) */
  titleId?: string
  /** Icon size */
  iconSize?: 'sm' | 'md' | 'lg'
  /** Icon tone */
  iconTone?: 'primary' | 'muted' | 'success' | 'warning' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
  /** Title size class */
  titleClassName?: string
  /** Subtitle size class */
  subtitleClassName?: string
  /** Additional CSS classes for the header wrapper */
  className?: string
  /** Use CardHeader wrapper (default: true) */
  useCardHeader?: boolean
  /** Alignment of items */
  align?: 'start' | 'center'
}

/**
 * IconCardHeader molecule - reusable card header with icon, title, and optional subtitle
 * Combines IconPlate, CardTitle, and CardDescription in a standard layout
 *
 * @example
 * // Standard card header
 * <IconCardHeader icon={<Database />} title="My Title" subtitle="Description" />
 *
 * // Large hero-style header
 * <IconCardHeader
 *   icon={<AlertTriangle />}
 *   title="GPU FAIL FAST policy"
 *   subtitle="No silent fallbacks..."
 *   titleClassName="text-3xl md:text-4xl font-extrabold"
 *   subtitleClassName="text-lg"
 *   useCardHeader={false}
 * />
 */
export function IconCardHeader({
  icon,
  title,
  subtitle,
  titleId,
  iconSize = 'lg',
  iconTone = 'primary',
  titleClassName = 'text-2xl',
  subtitleClassName,
  className,
  useCardHeader = true,
  align = 'center',
}: IconCardHeaderProps) {
  const content = (
    <div className={cn('flex gap-4', align === 'center' ? 'items-center' : 'items-start')}>
      <IconPlate icon={icon} size={iconSize} tone={iconTone} className="shrink-0" shape="rounded" />
      <div className="flex-1 min-w-0">
        {useCardHeader ? (
          <>
            <CardTitle id={titleId} className={titleClassName}>
              {title}
            </CardTitle>
            {subtitle && <CardDescription className={subtitleClassName}>{subtitle}</CardDescription>}
          </>
        ) : (
          <>
            <h3 id={titleId} className={cn('font-bold tracking-tight text-foreground mb-2', titleClassName)}>
              {title}
            </h3>
            {subtitle && <p className={cn('text-muted-foreground leading-relaxed', subtitleClassName)}>{subtitle}</p>}
          </>
        )}
      </div>
    </div>
  )

  if (useCardHeader) {
    return <CardHeader className={className}>{content}</CardHeader>
  }

  return <div className={className}>{content}</div>
}

export { IconCardHeader as default }
