import { cn } from "@rbee/ui/utils";

export interface FilterButtonProps {
  /** Button label */
  label: string;
  /** Whether this filter is active */
  active?: boolean;
  /** Click handler */
  onClick?: () => void;
  /** Aria label for accessibility */
  ariaLabel?: string;
  /** Optional className for custom styling */
  className?: string;
}

export function FilterButton({
  label,
  active = false,
  onClick,
  ariaLabel,
  className,
}: FilterButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      aria-pressed={active}
      className={cn(
        "px-2 py-1 text-xs transition-colors",
        active
          ? "rounded-md bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground",
        className
      )}
    >
      {label}
    </button>
  );
}
