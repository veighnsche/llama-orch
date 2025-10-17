import { BulletListItem } from '@rbee/ui/molecules'
import type { ReactNode } from 'react'

export interface CheckItemProps {
  /** Item content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
}

/**
 * @deprecated CheckItem is deprecated. Use BulletListItem with variant="check" and showPlate={false} instead.
 * This component is now a shim that wraps BulletListItem.
 * 
 * @example
 * // Instead of:
 * <CheckItem>Feature text</CheckItem>
 * 
 * // Use:
 * <BulletListItem variant="check" showPlate={false} title="Feature text" />
 */
export function CheckItem({ children, className }: CheckItemProps) {
  // Convert children to string for BulletListItem title prop
  const title = typeof children === 'string' ? children : String(children)
  
  return (
    <BulletListItem
      variant="check"
      showPlate={false}
      title={title}
      className={className}
    />
  )
}
