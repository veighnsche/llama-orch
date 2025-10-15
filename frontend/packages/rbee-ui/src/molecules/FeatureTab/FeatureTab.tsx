import { TabsTrigger } from "@rbee/ui/atoms/Tabs";
import { LucideIcon } from "lucide-react";

export interface FeatureTabProps {
  value: string;
  icon: LucideIcon;
  label: string;
  mobileLabel?: string;
}

/**
 * A feature tab trigger with icon and responsive text labels.
 * Used in tabbed feature sections to switch between different feature categories.
 */
export function FeatureTab({
  value,
  icon: Icon,
  label,
  mobileLabel,
}: FeatureTabProps) {
  return (
    <TabsTrigger
      value={value}
      className="flex flex-col sm:flex-row items-center justify-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg text-sm font-medium transition-colors"
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      <span className="hidden sm:inline">{label}</span>
      {mobileLabel && (
        <span className="text-xs text-muted-foreground block leading-none sm:hidden">
          {mobileLabel}
        </span>
      )}
    </TabsTrigger>
  );
}
