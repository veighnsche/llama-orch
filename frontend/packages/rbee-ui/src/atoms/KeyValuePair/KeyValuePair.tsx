import { cn } from "@rbee/ui/utils";

export interface KeyValuePairProps {
  /** The label/key text */
  label: string;
  /** The value text */
  value: string;
  /** Style variant for the value */
  valueVariant?:
    | "default"
    | "semibold"
    | "bold"
    | "success"
    | "warning"
    | "error";
  /** Additional CSS classes */
  className?: string;
}

const valueVariants = {
  default: "text-foreground",
  semibold: "font-semibold text-foreground",
  bold: "font-bold text-foreground",
  success: "font-bold text-chart-3",
  warning: "font-bold text-chart-1",
  error: "font-bold text-destructive",
};

export function KeyValuePair({
  label,
  value,
  valueVariant = "semibold",
  className,
}: KeyValuePairProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-6 text-sm",
        className
      )}
    >
      <span className="text-muted-foreground font-sans">{label}</span>
      <span className={cn(valueVariants[valueVariant])}>{value}</span>
    </div>
  );
}
