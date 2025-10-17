import { cn } from "@rbee/ui/utils";

export interface SegmentedControlOption {
  /** Unique key for the option */
  key: string;
  /** Display label */
  label: string;
}

export interface SegmentedControlProps {
  /** List of options */
  options: SegmentedControlOption[];
  /** Currently selected option key */
  value: string;
  /** Callback when selection changes */
  onChange: (key: string) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * SegmentedControl - iOS-style segmented control for switching between options
 *
 * @example
 * ```tsx
 * <SegmentedControl
 *   options={[
 *     { key: 'rbee', label: 'rbee' },
 *     { key: 'openai', label: 'OpenAI' },
 *   ]}
 *   value={selected}
 *   onChange={setSelected}
 * />
 * ```
 */
export function SegmentedControl({
  options,
  value,
  onChange,
  className,
}: SegmentedControlProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-center gap-2 rounded-lg border border-border bg-card/60 p-1",
        className
      )}
      role="tablist"
    >
      {options.map((option) => (
        <button
          key={option.key}
          onClick={() => onChange(option.key)}
          className={cn(
            "flex-1 rounded-md px-3 py-2 text-xs font-medium transition-colors",
            value === option.key
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
          role="tab"
          aria-selected={value === option.key}
          aria-controls={`panel-${option.key}`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
