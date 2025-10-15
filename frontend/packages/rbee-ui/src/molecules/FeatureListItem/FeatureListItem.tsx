import { cn } from "@rbee/ui/utils";
import { IconBox } from "@rbee/ui/molecules";
import type { LucideIcon } from "lucide-react";

export interface FeatureListItemProps {
  /** Lucide icon component */
  icon: LucideIcon;
  /** Feature title (bold part) */
  title: string;
  /** Feature description */
  description: string;
  /** Icon color variant */
  iconColor?:
    | "primary"
    | "chart-1"
    | "chart-2"
    | "chart-3"
    | "chart-4"
    | "chart-5";
  /** Icon variant */
  iconVariant?: "rounded" | "square";
  /** Icon size */
  iconSize?: "sm" | "md" | "lg";
  /** Additional CSS classes */
  className?: string;
}

export function FeatureListItem({
  icon,
  title,
  description,
  iconColor = "primary",
  iconVariant = "rounded",
  iconSize = "sm",
  className,
}: FeatureListItemProps) {
  return (
    <li className={cn("flex items-center gap-3", className)}>
      <IconBox
        icon={icon}
        size={iconSize}
        variant={iconVariant}
        color={iconColor}
      />
      <div className="text-base text-foreground">
        <strong className="font-semibold">{title}:</strong> {description}
      </div>
    </li>
  );
}
