import { Card, CardContent } from "@rbee/ui/atoms/Card";
import { cn } from "@rbee/ui/utils";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

// ──────────────────────────────────────────────────────────────────────────────
// Variants
// ──────────────────────────────────────────────────────────────────────────────

const featureInfoCardVariants = cva(
  "group transition-all animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none",
  {
    variants: {
      tone: {
        default: "bg-card hover:bg-card/80 hover:border-primary/50",
        neutral: "bg-background hover:bg-muted/30 hover:border-primary/50",
        primary:
          "border border-primary/40 bg-gradient-to-b from-primary/15 to-background backdrop-blur supports-[backdrop-filter]:bg-background/60 hover:border-primary/50",
        destructive:
          "border border-destructive/40 bg-gradient-to-b from-destructive/15 to-background backdrop-blur supports-[backdrop-filter]:bg-background/60 hover:border-primary/50",
        muted:
          "border-muted bg-gradient-to-b from-muted/50 to-background backdrop-blur supports-[backdrop-filter]:bg-background/60 hover:border-primary/50",
        chart2: "bg-card hover:bg-card/80 hover:border-primary/50",
        chart3: "bg-card hover:bg-card/80 hover:border-primary/50",
      },
      showBorder: {
        true: "border border-border",
        false: "",
      },
    },
    defaultVariants: {
      tone: "default",
      showBorder: false,
    },
  }
);

const iconContainerVariants = cva(
  "flex items-center justify-center rounded-xl",
  {
    variants: {
      tone: {
        default: "bg-primary/10",
        neutral: "bg-primary/10",
        primary: "bg-primary/10",
        destructive: "bg-destructive/10",
        muted: "bg-muted",
        chart2: "bg-chart-2/10",
        chart3: "bg-chart-3/10",
      },
      variant: {
        default: "mb-4 h-11 w-11",
        compact: "mb-3 h-8 w-8",
      },
    },
    defaultVariants: {
      tone: "default",
      variant: "default",
    },
  }
);

const iconVariants = cva("", {
  variants: {
    tone: {
      default: "text-primary",
      neutral: "text-primary",
      primary: "text-primary",
      destructive: "text-destructive",
      muted: "text-muted-foreground",
      chart2: "text-chart-2",
      chart3: "text-chart-3",
    },
    variant: {
      default: "h-6 w-6",
      compact: "h-4 w-4",
    },
  },
  defaultVariants: {
    tone: "default",
    variant: "default",
  },
});

const tagVariants = cva(
  "mt-3 inline-flex rounded-full px-2.5 py-1 text-xs tabular-nums",
  {
    variants: {
      tone: {
        default: "bg-muted text-muted-foreground",
        neutral: "bg-muted text-muted-foreground",
        primary: "bg-primary/10 text-primary",
        destructive: "bg-destructive/10 text-destructive",
        muted: "bg-muted text-muted-foreground",
        chart2: "bg-chart-2/10 text-chart-2",
        chart3: "bg-chart-3/10 text-chart-3",
      },
    },
    defaultVariants: {
      tone: "default",
    },
  }
);

const contentPaddingVariants = cva("", {
  variants: {
    variant: {
      default: "p-6 sm:p-7",
      compact: "p-4",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

const titleVariants = cva("font-semibold text-card-foreground", {
  variants: {
    variant: {
      default: "mb-2 text-lg",
      compact: "mb-1 text-sm",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

const bodyVariants = cva("text-balance leading-relaxed text-muted-foreground", {
  variants: {
    size: {
      sm: "text-sm",
      base: "text-base",
    },
    variant: {
      default: "",
      compact: "text-xs",
    },
  },
  defaultVariants: {
    size: "sm",
    variant: "default",
  },
});

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/**
 * FeatureInfoCard displays an icon, title, and body text.
 * Used for benefits, problems, features, and solution cards.
 *
 * @example
 * ```tsx
 * <FeatureInfoCard
 *   icon={<DollarSign className="h-6 w-6" />}
 *   title="Zero ongoing costs"
 *   body="Pay only for electricity. No API bills, no per-token surprises."
 *   tone="primary"
 *   size="sm"
 * />
 * ```
 */
export interface FeatureInfoCardProps
  extends VariantProps<typeof featureInfoCardVariants>,
    VariantProps<typeof bodyVariants> {
  /** Icon element or component */
  icon: React.ComponentType<{ className?: string }> | React.ReactNode;
  /** Card title */
  title: string;
  /** Card body text */
  body: string;
  /** Optional tag/badge text (e.g., "Loss €50/mo") */
  tag?: string;
  /** Additional CSS classes */
  className?: string;
  /** Animation delay class (e.g., "delay-75") */
  delay?: string;
  /** Show border (default: true) */
  showBorder?: boolean;
  /** Variant: default (full size) or compact (smaller, inline icon) */
  variant?: "default" | "compact";
}

// ──────────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────────

export function FeatureInfoCard({
  icon,
  title,
  body,
  tag,
  tone,
  size,
  className,
  delay,
  showBorder,
  variant = "default",
}: FeatureInfoCardProps) {
  // Handle both icon types (Component or ReactNode)
  let IconComponent: React.ComponentType<{ className?: string }> | null = null;

  if (typeof icon === "function") {
    IconComponent = icon as React.ComponentType<{ className?: string }>;
  }

  return (
    <Card
      className={cn(
        featureInfoCardVariants({ tone, showBorder }),
        delay,
        className
      )}
    >
      <CardContent className={contentPaddingVariants({ variant })}>
        {/* Icon */}
        <div className={iconContainerVariants({ tone, variant })} aria-hidden="true">
          {IconComponent ? (
            <IconComponent className={iconVariants({ tone, variant })} />
          ) : React.isValidElement(icon) ? (
            React.cloneElement(icon, {
              // @ts-expect-error - icon className merging
              className: cn(icon.props.className, iconVariants({ tone, variant })),
            })
          ) : null}
        </div>

        {/* Title */}
        <h3 className={titleVariants({ variant })}>
          {title}
        </h3>

        {/* Body */}
        <p className={bodyVariants({ size, variant })}>{body}</p>

        {/* Optional Tag */}
        {tag && <span className={tagVariants({ tone })}>{tag}</span>}
      </CardContent>
    </Card>
  );
}

export {
  featureInfoCardVariants,
  iconContainerVariants,
  iconVariants,
  tagVariants,
  contentPaddingVariants,
  titleVariants,
  bodyVariants,
};
