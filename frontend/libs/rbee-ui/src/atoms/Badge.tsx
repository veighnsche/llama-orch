import * as React from 'react';
import { cn } from '../utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'destructive';
}

/**
 * Badge component - Small status or label indicator
 * 
 * @example
 * ```tsx
 * <Badge>New</Badge>
 * <Badge variant="secondary">Beta</Badge>
 * ```
 */
export const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold',
          'transition-colors focus:outline-none focus:ring-2',
          'focus:ring-[var(--rbee-ring)] focus:ring-offset-2',
          {
            'bg-[var(--rbee-primary)] text-[var(--rbee-primary-foreground)]':
              variant === 'default',
            'bg-[var(--rbee-secondary)] text-[var(--rbee-secondary-foreground)]':
              variant === 'secondary',
            'border border-[var(--rbee-border)] text-[var(--rbee-foreground)]':
              variant === 'outline',
            'bg-[var(--rbee-destructive)] text-[var(--rbee-destructive-foreground)]':
              variant === 'destructive',
          },
          className
        )}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';
