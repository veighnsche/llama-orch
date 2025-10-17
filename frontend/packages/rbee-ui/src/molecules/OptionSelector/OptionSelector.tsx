import { cn } from '@rbee/ui/utils'

export interface Option<T = any> {
  /** Unique identifier for the option */
  id: string
  /** Display label */
  label: string
  /** Optional subtitle/description */
  subtitle?: string
  /** Data payload for this option */
  data: T
}

export interface OptionSelectorProps<T = any> {
  /** List of options */
  options: Option<T>[]
  /** Callback when an option is selected */
  onSelect: (data: T) => void
  /** Number of columns in the grid */
  columns?: 2 | 3 | 4
  /** Additional CSS classes */
  className?: string
}

/**
 * OptionSelector molecule - displays a grid of option buttons for quick selection
 *
 * @example
 * ```tsx
 * <OptionSelector
 *   options={[
 *     { id: 'casual', label: 'Casual', subtitle: '8h • 50%', data: { hours: 8, util: 50 } },
 *     { id: 'daily', label: 'Daily', subtitle: '16h • 70%', data: { hours: 16, util: 70 } },
 *   ]}
 *   onSelect={(data) => console.log(data)}
 *   columns={3}
 * />
 * ```
 */
export function OptionSelector<T = any>({ options, onSelect, columns = 3, className }: OptionSelectorProps<T>) {
  const gridCols = {
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4',
  }

  return (
    <div className={cn('grid gap-2 font-sans', gridCols[columns], className)}>
      {options.map((option) => (
        <button
          key={option.id}
          onClick={() => onSelect(option.data)}
          className="rounded-md border border-border bg-background/60 px-3 py-2 text-xs transition-colors hover:bg-background"
        >
          <div className="font-medium">{option.label}</div>
          {option.subtitle && <div className="text-muted-foreground">{option.subtitle}</div>}
        </button>
      ))}
    </div>
  )
}

export { OptionSelector as default }
