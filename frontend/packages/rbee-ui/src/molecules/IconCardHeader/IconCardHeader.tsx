import { CardHeader, CardTitle, CardDescription } from "@rbee/ui/atoms";
import { IconPlate } from "@rbee/ui/molecules";
import type { LucideIcon } from "lucide-react";

export interface IconCardHeaderProps {
  /** Lucide icon component */
  icon: LucideIcon;
  /** Card title */
  title: string;
  /** Optional subtitle/description */
  subtitle?: string;
  /** ID for the title (for aria-labelledby) */
  titleId?: string;
  /** Icon size */
  iconSize?: "sm" | "md" | "lg";
  /** Icon tone */
  iconTone?: "primary" | "muted" | "success" | "warning";
  /** Title size class */
  titleClassName?: string;
  /** Additional CSS classes for the header wrapper */
  className?: string;
}

/**
 * IconCardHeader molecule - reusable card header with icon, title, and optional subtitle
 * Combines IconPlate, CardTitle, and CardDescription in a standard layout
 */
export function IconCardHeader({
  icon,
  title,
  subtitle,
  titleId,
  iconSize = "lg",
  iconTone = "primary",
  titleClassName = "text-2xl",
  className,
}: IconCardHeaderProps) {
  return (
    <CardHeader className={className}>
      <div className="flex items-center gap-3">
        <IconPlate
          icon={icon}
          size={iconSize}
          tone={iconTone}
          className="shrink-0"
          shape="rounded"
        />
        <div>
          <CardTitle id={titleId} className={titleClassName}>
            {title}
          </CardTitle>
          {subtitle && <CardDescription>{subtitle}</CardDescription>}
        </div>
      </div>
    </CardHeader>
  );
}
