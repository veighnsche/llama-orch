import { cn } from "@rbee/ui/utils";

export interface BulletListItemProps {
  /** Item title */
  title: string;
  /** Optional description */
  description?: string;
  /** Optional meta text (right-aligned) */
  meta?: string;
  /** Bullet color (Tailwind class) */
  color?: string;
  /** Bullet variant */
  variant?: "dot" | "check" | "arrow";
  /** Additional CSS classes */
  className?: string;
}

export function BulletListItem({
  title,
  description,
  meta,
  color = "chart-3",
  variant = "dot",
  className,
}: BulletListItemProps) {
  const colorClasses = {
    primary: {
      bg: "bg-primary/20",
      bgSolid: "bg-primary",
      text: "text-primary",
    },
    "chart-1": {
      bg: "bg-chart-1/20",
      bgSolid: "bg-chart-1",
      text: "text-chart-1",
    },
    "chart-2": {
      bg: "bg-chart-2/20",
      bgSolid: "bg-chart-2",
      text: "text-chart-2",
    },
    "chart-3": {
      bg: "bg-chart-3/20",
      bgSolid: "bg-chart-3",
      text: "text-chart-3",
    },
    "chart-4": {
      bg: "bg-chart-4/20",
      bgSolid: "bg-chart-4",
      text: "text-chart-4",
    },
    "chart-5": {
      bg: "bg-chart-5/20",
      bgSolid: "bg-chart-5",
      text: "text-chart-5",
    },
  };

  const colors =
    colorClasses[color as keyof typeof colorClasses] || colorClasses["chart-3"];

  const renderBullet = () => {
    switch (variant) {
      case "dot":
        return (
          <div
            className={cn(
              "h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5",
              colors.bg
            )}
          >
            <div className={cn("h-2 w-2 rounded-full", colors.bgSolid)}></div>
          </div>
        );
      case "check":
        return (
          <div
            className={cn(
              "h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 -translate-y-[4px]",
              colors.bg,
              colors.text
            )}
          >
            ✓
          </div>
        );
      case "arrow":
        return (
          <div
            className={cn(
              "h-6 w-6 flex items-center justify-center flex-shrink-0 mt-0.5",
              colors.text
            )}
          >
            →
          </div>
        );
    }
  };

  return (
    <li className={cn("flex items-start gap-3", className)}>
      {renderBullet()}
      <div className="flex-1">
        <div className="flex items-center justify-between gap-2">
          <div className="font-medium text-foreground">{title}</div>
          {meta && (
            /** please do not remove the translate token so that I can keep watching if the custom values work */
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
