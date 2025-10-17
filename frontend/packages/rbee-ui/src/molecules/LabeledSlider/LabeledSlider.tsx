import { Slider } from '@rbee/ui/atoms/Slider'
import { cn } from '@rbee/ui/utils'

export interface LabeledSliderProps {
  /** Label text */
  label: string
  /** Current value (array for Slider component) */
  value: number[]
  /** Callback when value changes */
  onValueChange: (value: number[]) => void
  /** Minimum value */
  min: number
  /** Maximum value */
  max: number
  /** Step increment */
  step?: number
  /** Aria label for accessibility */
  ariaLabel: string
  /** Format function for the displayed value */
  formatValue: (value: number) => string
  /** Optional min label (displayed below slider) */
  minLabel?: string
  /** Optional max label (displayed below slider) */
  maxLabel?: string
  /** Optional helper text below the slider */
  helperText?: string | ((value: number) => string)
  /** Additional CSS classes */
  className?: string
}

/**
 * LabeledSlider molecule - a slider with label, value display, and optional helper text
 *
 * @example
 * ```tsx
 * <LabeledSlider
 *   label="Hours Per Day"
 *   value={[20]}
 *   onValueChange={setHours}
 *   min={1}
 *   max={24}
 *   step={1}
 *   ariaLabel="Hours available per day"
 *   formatValue={(v) => `${v}h`}
 *   minLabel="1h"
 *   maxLabel="24h"
 *   helperText={(v) => `≈ ${v * 30}h / mo`}
 * />
 * ```
 */
export function LabeledSlider({
  label,
  value,
  onValueChange,
  min,
  max,
  step = 1,
  ariaLabel,
  formatValue,
  minLabel,
  maxLabel,
  helperText,
  className,
}: LabeledSliderProps) {
  const currentValue = value[0]
  const resolvedHelperText = typeof helperText === 'function' ? helperText(currentValue) : helperText

  return (
    <div className={cn('font-sans', className)}>
      <div className="mb-3 flex items-center justify-between">
        <label className="text-sm font-medium text-muted-foreground">{label}</label>
        <span className="tabular-nums text-lg font-bold text-primary font-serif">{formatValue(currentValue)}</span>
      </div>
      <Slider
        value={value}
        onValueChange={onValueChange}
        min={min}
        max={max}
        step={step}
        aria-label={ariaLabel}
        className="[&_[role=slider]]:bg-primary"
      />
      {(minLabel || maxLabel) && (
        <div className="mt-2 flex justify-between text-xs text-muted-foreground">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      )}
      {helperText && <div className="mt-1 text-xs text-muted-foreground">{resolvedHelperText}</div>}
    </div>
  )
}

export { LabeledSlider as default }
