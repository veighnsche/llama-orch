import { cn } from "@rbee/ui/utils";
import { cva, type VariantProps } from "class-variance-authority";
import { Check } from "lucide-react";

const bulletListItemVariants = cva("flex items-start gap-3", {
  variants: {
    color: {
      primary: "",
      "chart-1": "",
      "chart-2": "",
      "chart-3": "",
      "chart-4": "",
      "chart-5": "",
    },
  },
  defaultVariants: {
    color: "chart-3",
  },
});

const bulletContainerVariants = cva(
  "h-6 w-6 flex items-center justify-center flex-shrink-0 mt-0.5",
  {
    variants: {
      color: {
        primary: "",
        "chart-1": "",
        "chart-2": "",
        "chart-3": "",
        "chart-4": "",
        "chart-5": "",
      },
      variant: {
        dot: "rounded-full -translate-y-[3px]",
        "dot-plain": "-translate-y-[3px]",
        check: "rounded-full -translate-y-[4px]",
        "check-plain": "-translate-y-[4px]",
        arrow: "-translate-y-[4px]",
        "arrow-plain": "-translate-y-[4px]",
      },
    },
    compoundVariants: [
      // Background variants for dot and check
      { variant: "dot", color: "primary", class: "bg-primary/20" },
      { variant: "dot", color: "chart-1", class: "bg-chart-1/20" },
      { variant: "dot", color: "chart-2", class: "bg-chart-2/20" },
      { variant: "dot", color: "chart-3", class: "bg-chart-3/20" },
      { variant: "dot", color: "chart-4", class: "bg-chart-4/20" },
      { variant: "dot", color: "chart-5", class: "bg-chart-5/20" },
      {
        variant: "check",
        color: "primary",
        class: "bg-primary/20 text-primary",
      },
      {
        variant: "check",
        color: "chart-1",
        class: "bg-chart-1/20 text-chart-1",
      },
      {
        variant: "check",
        color: "chart-2",
        class: "bg-chart-2/20 text-chart-2",
      },
      {
        variant: "check",
        color: "chart-3",
        class: "bg-chart-3/20 text-chart-3",
      },
      {
        variant: "check",
        color: "chart-4",
        class: "bg-chart-4/20 text-chart-4",
      },
      {
        variant: "check",
        color: "chart-5",
        class: "bg-chart-5/20 text-chart-5",
      },
      // Plain check variants (no background)
      { variant: "check-plain", color: "primary", class: "text-primary" },
      { variant: "check-plain", color: "chart-1", class: "text-chart-1" },
      { variant: "check-plain", color: "chart-2", class: "text-chart-2" },
      { variant: "check-plain", color: "chart-3", class: "text-chart-3" },
      { variant: "check-plain", color: "chart-4", class: "text-chart-4" },
      { variant: "check-plain", color: "chart-5", class: "text-chart-5" },
      // Text color variants for arrow
      { variant: "arrow", color: "primary", class: "text-primary" },
      { variant: "arrow", color: "chart-1", class: "text-chart-1" },
      { variant: "arrow", color: "chart-2", class: "text-chart-2" },
      { variant: "arrow", color: "chart-3", class: "text-chart-3" },
      { variant: "arrow", color: "chart-4", class: "text-chart-4" },
      { variant: "arrow", color: "chart-5", class: "text-chart-5" },
      // Plain arrow variants (same as regular arrow, kept for consistency)
      { variant: "arrow-plain", color: "primary", class: "text-primary" },
      { variant: "arrow-plain", color: "chart-1", class: "text-chart-1" },
      { variant: "arrow-plain", color: "chart-2", class: "text-chart-2" },
      { variant: "arrow-plain", color: "chart-3", class: "text-chart-3" },
      { variant: "arrow-plain", color: "chart-4", class: "text-chart-4" },
      { variant: "arrow-plain", color: "chart-5", class: "text-chart-5" },
    ],
    defaultVariants: {
      color: "chart-3",
      variant: "dot",
    },
  }
);

const bulletDotVariants = cva("h-2 w-2 rounded-full", {
  variants: {
    color: {
      primary: "bg-primary",
      "chart-1": "bg-chart-1",
      "chart-2": "bg-chart-2",
      "chart-3": "bg-chart-3",
      "chart-4": "bg-chart-4",
      "chart-5": "bg-chart-5",
    },
  },
  defaultVariants: {
    color: "chart-3",
  },
});

export interface BulletListItemProps
  extends VariantProps<typeof bulletListItemVariants> {
  /** Item title */
  title: string;
  /** Optional description */
  description?: string;
  /** Optional meta text (right-aligned) */
  meta?: string;
  /** Bullet variant */
  variant?: "dot" | "check" | "arrow";
  /** Show background plate for check/arrow (default: true) */
  showPlate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function BulletListItem({
  title,
  description,
  meta,
  color = "chart-3",
  variant = "dot",
  showPlate = true,
  className,
}: BulletListItemProps) {
  const renderBullet = () => {
    const effectiveVariant =
      (variant === "check" || variant === "arrow" || variant === "dot") && !showPlate
        ? (`${variant}-plain` as const)
        : variant;

    switch (variant) {
      case "dot":
        return (
          <div className={bulletContainerVariants({ color, variant: effectiveVariant })}>
            <div className={bulletDotVariants({ color })}></div>
          </div>
        );
      case "check":
        return (
          <div
            className={bulletContainerVariants({
              color,
              variant: effectiveVariant,
            })}
          >
            <Check className="h-4 w-4" aria-hidden="true" />
          </div>
        );
      case "arrow":
        return (
          <div
            className={bulletContainerVariants({
              color,
              variant: effectiveVariant,
            })}
          >
            →
          </div>
        );
    }
  };

  return (
    <li className={cn(bulletListItemVariants({ color }), className)}>
      {renderBullet()}
      <div className="flex-1">
        <div className="flex items-center justify-between gap-2">
          <div className="font-medium text-foreground">{title}</div>
          {meta && (
            <div className="text-xs text-muted-foreground whitespace-nowrap font-sans translate-y-[2px]">
              {meta}
            </div>
          )}
        </div>
        {description && (
          <div className="text-sm text-muted-foreground font-sans">
            {description}
          </div>
        )}
      </div>
    </li>
  );
}
