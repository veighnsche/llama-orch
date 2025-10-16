import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

export interface FeatureListItemProps {
  /** Rendered icon component */
  icon: React.ReactNode
  /** Feature title (bold part) */
  title: string
  /** Feature description */
  description: string
  /** Icon color variant */
  iconColor?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
  /** Icon variant */
  iconVariant?: 'rounded' | 'square' | 'circle'
  /** Icon size */
  iconSize?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
}

export function FeatureListItem({
  icon,
  title,
  description,
  iconColor = 'primary',
  iconVariant = 'rounded',
  iconSize = 'sm',
  className,
}: FeatureListItemProps) {
  return (
    <li className={cn('flex items-center gap-3', className)}>
      <IconPlate icon={icon} size={iconSize} shape={iconVariant} tone={iconColor} className="flex-shrink-0" />
      <div className="text-base text-foreground">
        <strong className="font-semibold">{title}:</strong> {description}
      </div>
    </li>
  )
}
