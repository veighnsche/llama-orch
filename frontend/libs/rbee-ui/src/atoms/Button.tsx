import * as React from 'react';
import { cn } from '../utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg' | 'icon';
  asChild?: boolean;
}

/**
 * Button component - Primary interactive element
 * 
 * @example
 * ```tsx
 * <Button variant="default" size="md">Click me</Button>
 * <Button variant="outline">Secondary action</Button>
 * ```
 */
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'md', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          // Base styles
          'inline-flex items-center justify-center gap-2 rounded-lg font-medium',
          'transition-colors focus-visible:outline-none focus-visible:ring-2',
          'focus-visible:ring-[var(--rbee-ring)] focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          
          // Variants
          {
            'bg-[var(--rbee-primary)] text-[var(--rbee-primary-foreground)] hover:opacity-90':
              variant === 'default',
            'bg-[var(--rbee-secondary)] text-[var(--rbee-secondary-foreground)] hover:bg-[var(--rbee-secondary)]/80':
              variant === 'secondary',
            'border border-[var(--rbee-border)] bg-transparent hover:bg-[var(--rbee-accent)] hover:text-[var(--rbee-accent-foreground)]':
              variant === 'outline',
            'hover:bg-[var(--rbee-accent)] hover:text-[var(--rbee-accent-foreground)]':
              variant === 'ghost',
            'bg-[var(--rbee-destructive)] text-[var(--rbee-destructive-foreground)] hover:opacity-90':
              variant === 'destructive',
          },
          
          // Sizes
          {
            'h-9 px-3 text-sm': size === 'sm',
            'h-10 px-4 text-base': size === 'md',
            'h-11 px-6 text-lg': size === 'lg',
            'h-10 w-10 p-0': size === 'icon',
          },
          
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = 'Button';
