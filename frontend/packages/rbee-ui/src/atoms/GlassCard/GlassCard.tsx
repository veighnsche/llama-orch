import { cn } from "@rbee/ui/utils";
import type { HTMLAttributes, ReactNode } from "react";

export interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  /** Card content */
  children: ReactNode;
  /** Additional CSS classes */
  className?: string;
}

/**
 * GlassCard - A card with frosted glass effect (backdrop blur + semi-transparent background)
 *
 * @example
 * ```tsx
 * <GlassCard>
 *   <p>Content with glass effect</p>
 * </GlassCard>
 * ```
 */
export function GlassCard({ children, className, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl backdrop-blur-md bg-secondary/60 dark:bg-secondary/30 shadow-lg",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
