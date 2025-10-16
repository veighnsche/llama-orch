import { cn } from "@rbee/ui/utils";

export interface CrateCardProps {
  /** Crate name */
  name: string;
  /** Crate description */
  description: string;
  /** Hover border color class */
  hoverColor?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * CrateCard molecule - lightweight card for displaying crate/package information
 * Used in security crate lattices and similar compact grid layouts
 *
 * @example
 * <CrateCard
 *   name="auth-min"
 *   description="Timing-safe tokens, zero-trust auth."
 *   hoverColor="hover:border-chart-2/50"
 * />
 */
export function CrateCard({
  name,
  description,
  hoverColor = "hover:border-primary/50",
  className,
}: CrateCardProps) {
  return (
    <div
      className={cn(
        "group rounded-lg bg-background border border-border p-4 transition-colors",
        hoverColor,
        className
      )}
    >
      <div className="font-semibold text-foreground mb-1">{name}</div>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
