import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export interface TimelineStepProps {
  /** Timestamp or step label */
  timestamp: string;
  /** Step title/heading */
  title: ReactNode;
  /** Step description */
  description?: string;
  /** Additional CSS classes */
  className?: string;
  /** Variant for different visual styles */
  variant?: "default" | "success" | "warning" | "error";
}

/**
 * TimelineStep molecule - displays a single step in a timeline or sequence
 * Commonly used for process flows, cancellation sequences, or step-by-step guides
 *
 * @example
 * <TimelineStep
 *   timestamp="t+0ms"
 *   title="Client sends POST /v1/cancel"
 *   description="Idempotent request."
 * />
 *
 * @example
 * <TimelineStep
 *   timestamp="t+120ms"
 *   title={<span className="text-chart-3">Worker idle ✓</span>}
 *   description="Ready for next task."
 *   variant="success"
 * />
 */
export function TimelineStep({
  timestamp,
  title,
  description,
  className,
  variant = "default",
}: TimelineStepProps) {
  const variantClasses = {
    default: "",
    success: "border-chart-3/30",
    warning: "border-warning/30",
    error: "border-destructive/30",
  };

  return (
    <div
      className={cn(
        "bg-background border border-border rounded-xl p-4 hover:ring-1 hover:ring-border transition-all font-sans",
        variantClasses[variant],
        className
      )}
    >
      <div className="text-xs text-muted-foreground">{timestamp}</div>
      <div className="font-semibold text-foreground mt-1">{title}</div>
      {description && (
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      )}
    </div>
  );
}
