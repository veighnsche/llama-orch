import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export interface LegendItem {
  /** Icon component */
  icon: ReactNode;
  /** Label text */
  label: string;
}

export interface LegendProps {
  /** Legend items with icons */
  items?: LegendItem[];
  /** Additional note text */
  note?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Legend - Display legend items with icons and optional note
 *
 * @example
 * ```tsx
 * <Legend
 *   items={[
 *     { icon: <Check className="h-3.5 w-3.5" />, label: 'Available' },
 *     { icon: <X className="h-3.5 w-3.5" />, label: 'Not available' },
 *   ]}
 *   note="Partial = limited coverage"
 * />
 * ```
 */
export function Legend({ items, note, className }: LegendProps) {
  if (!items && !note) {
    return null;
  }

  return (
    <div
      className={cn(
        "text-xs text-muted-foreground flex flex-wrap gap-4 justify-center",
        className
      )}
    >
      {items?.map((item, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {item.icon}
          {item.label}
        </span>
      ))}
      {note && <span>{note}</span>}
    </div>
  );
}
