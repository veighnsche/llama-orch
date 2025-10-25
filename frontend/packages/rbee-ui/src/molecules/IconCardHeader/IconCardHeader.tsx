import { CardHeader } from '@rbee/ui/atoms'
import { IconPlate } from '@rbee/ui/molecules'
import { cn, parseInlineMarkdown } from '@rbee/ui/utils'
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
  /** Alignment of items */
  align?: 'start' | 'center'
}

/**
 * IconCardHeader molecule - reusable card header with icon, title, and optional subtitle
 * Always wraps content in CardHeader with CardTitle and CardDescription
 *
 * @example
 * <Card className="p-8">
 *   <IconCardHeader
 *     icon={<Database />}
 *     title="My Title"
 *     subtitle="Description"
 *   />
 *   <CardContent className="p-0">...</CardContent>
 * </Card>
 */
export function IconCardHeader({
  icon,
  title,
  subtitle,
  iconSize = 'lg',
  iconTone = 'primary',
  className,
  align = 'start',
}: IconCardHeaderProps) {
  return (
    <CardHeader className={className}>
      <div className={cn('flex gap-4', align === 'center' ? 'items-center' : 'items-start')}>
        <IconPlate
          icon={icon}
          size={iconSize}
          tone={iconTone}
          className={cn('shrink-0', iconSize === 'sm' ? '' : 'ranslate-y-1')}
          shape="rounded"
        />
        <div className="flex-1">
          <h3 className="text-xl font-semibold text-card-foreground mb-1">{title}</h3>
          {subtitle && <p className="text-sm text-muted-foreground">{parseInlineMarkdown(subtitle)}</p>}
        </div>
      </div>
    </CardHeader>
  )
}

export { IconCardHeader as default }
