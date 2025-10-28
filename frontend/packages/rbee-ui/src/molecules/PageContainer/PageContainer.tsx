import { ScrollArea } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'
import { HelperTextItem, type HelperTextItemProps } from '../HelperTextItem'

export interface PageContainerProps {
  /** Page title */
  title: string
  /** Optional page description */
  description?: string
  /** Optional actions (buttons, etc.) aligned to the right */
  actions?: ReactNode
  /** Page content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
  /** Content spacing variant */
  spacing?: 'default' | 'compact' | 'relaxed'
  /** Padding variant (default: 'default' = p-4) */
  padding?: 'none' | 'sm' | 'default' | 'lg'
  /** Optional helper text items shown at the bottom of the page */
  helperText?: HelperTextItemProps[]
}

const spacingClasses = {
  compact: 'space-y-3',
  default: 'space-y-4',
  relaxed: 'space-y-6',
} as const

const paddingClasses = {
  none: '',
  sm: 'p-2',
  default: 'p-4',
  lg: 'p-6',
} as const

/**
 * PageContainer - Consistent page wrapper for web-ui app pages
 *
 * Provides:
 * - Consistent page structure (flex-1 container)
 * - Title + description header
 * - Optional action buttons
 * - Proper spacing between sections
 *
 * @example
 * ```tsx
 * <PageContainer
 *   title="Dashboard"
 *   description="Monitor your queen, hives, workers, and models"
 * >
 *   <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
 *     {/* Page content *\/}
 *   </div>
 * </PageContainer>
 * ```
 */
export function PageContainer({
  title,
  description,
  actions,
  children,
  className,
  spacing = 'default',
  padding = 'default',
  helperText,
}: PageContainerProps) {
  return (
    <ScrollArea className="h-full">
      <div className={cn(paddingClasses[padding], className)}>
        {/* Header */}
        <div className={cn('flex items-start justify-between gap-4 mb-6', actions ? 'flex-wrap sm:flex-nowrap' : '')}>
          <div className="min-w-0 flex-1">
            <h1 className="text-3xl font-bold tracking-tight text-foreground">{title}</h1>
            {description && <p className="text-muted-foreground mt-1">{description}</p>}
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </div>

        {/* Content */}
        <div className={cn(spacingClasses[spacing])}>{children}</div>

        {/* Helper Text */}
        {helperText && helperText.length > 0 && (
          <div className="mt-8 pt-6 border-t border-border/50 space-y-3">
            {helperText.map((item, index) => (
              <HelperTextItem key={index} title={item.title} description={item.description} />
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  )
}
