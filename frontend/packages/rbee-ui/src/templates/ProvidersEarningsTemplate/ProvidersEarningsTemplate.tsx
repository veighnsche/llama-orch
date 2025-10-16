'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { Slider } from '@rbee/ui/atoms/Slider'
import { cn } from '@rbee/ui/utils'
import { Cpu } from 'lucide-react'
import { useRef, useState } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersEarningsGPUModel = {
  name: string
  baseRate: number
  vram: number
}

export type ProvidersEarningsPreset = {
  label: string
  hours: number
  utilization: number
}

export type ProvidersEarningsTemplateProps = {
  gpuModels: ProvidersEarningsGPUModel[]
  presets: ProvidersEarningsPreset[]
  commission: number
  configTitle: string
  selectGPULabel: string
  presetsLabel: string
  hoursLabel: string
  utilizationLabel: string
  earningsTitle: string
  monthlyLabel: string
  basedOnText: (hours: number, utilization: number) => string
  takeHomeLabel: string
  dailyLabel: string
  yearlyLabel: string
  breakdownTitle: string
  hourlyRateLabel: string
  hoursPerMonthLabel: string
  utilizationBreakdownLabel: string
  commissionLabel: string
  yourTakeHomeLabel: string
  ctaLabel: string
  ctaAriaLabel: string
  secondaryCTALabel: string
  disclaimerText: string
  formatCurrency: (n: number, opts?: Intl.NumberFormatOptions) => string
  formatHourly: (n: number) => string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersEarningsTemplate - Interactive earnings calculator for GPU providers
 *
 * @example
 * ```tsx
 * <ProvidersEarningsTemplate
 *   kicker="Estimate Your Earnings"
 *   title="Calculate Your Potential Earnings"
 *   gpuModels={[...]}
 *   presets={[...]}
 *   commission={0.15}
 *   formatCurrency={(n) => `€${n}`}
 *   // ... other props
 * />
 * ```
 */
export function ProvidersEarningsTemplate({
  gpuModels,
  presets,
  commission,
  configTitle,
  selectGPULabel,
  presetsLabel,
  hoursLabel,
  utilizationLabel,
  earningsTitle,
  monthlyLabel,
  basedOnText,
  takeHomeLabel,
  dailyLabel,
  yearlyLabel,
  breakdownTitle,
  hourlyRateLabel,
  hoursPerMonthLabel,
  utilizationBreakdownLabel,
  commissionLabel,
  yourTakeHomeLabel,
  ctaLabel,
  ctaAriaLabel,
  secondaryCTALabel,
  disclaimerText,
  formatCurrency,
  formatHourly,
}: ProvidersEarningsTemplateProps) {
  const [selectedGPU, setSelectedGPU] = useState(gpuModels[0])
  const [utilization, setUtilization] = useState([80])
  const [hoursPerDay, setHoursPerDay] = useState([20])
  const gpuListRef = useRef<HTMLDivElement>(null)

  const applyPreset = (hours: number, util: number) => {
    setHoursPerDay([hours])
    setUtilization([util])
  }

  const hourlyRate = selectedGPU.baseRate
  const dailyEarnings = hourlyRate * hoursPerDay[0] * (utilization[0] / 100)
  const monthlyEarnings = dailyEarnings * 30
  const yearlyEarnings = monthlyEarnings * 12
  const takeHome = monthlyEarnings * (1 - commission)

  return (
    <div id="earnings-calculator">
      <div className="mx-auto max-w-4xl">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Calculator Inputs */}
          <div className="animate-in fade-in slide-in-from-bottom-2 rounded-2xl border bg-gradient-to-b from-card to-background p-6 motion-reduce:animate-none sm:p-8">
            <div className="mb-6 flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />
              <h3 className="text-xl font-bold text-foreground">{configTitle}</h3>
            </div>

            <div className="space-y-6">
              {/* GPU Selection */}
              <div ref={gpuListRef}>
                <label className="mb-3 block text-sm font-medium text-muted-foreground">{selectGPULabel}</label>
                <div
                  role="radiogroup"
                  aria-label="GPU model selection"
                  className="grid max-h-[320px] gap-2 overflow-y-auto pr-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground"
                >
                  {gpuModels.map((gpu) => {
                    const isSelected = selectedGPU.name === gpu.name
                    return (
                      <button
                        key={gpu.name}
                        role="radio"
                        aria-checked={isSelected}
                        onClick={() => setSelectedGPU(gpu)}
                        className={cn(
                          'relative min-w-0 rounded-lg border p-3 text-left transition-transform hover:translate-y-0.5',
                          isSelected
                            ? 'animate-in zoom-in-95 border-primary bg-primary/10 before:absolute before:bottom-0 before:left-0 before:top-0 before:w-0.5 before:bg-primary motion-reduce:animate-none'
                            : 'border-border bg-background/50 hover:border-border/70',
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="min-w-0">
                            <div className="font-medium text-foreground">{gpu.name}</div>
                            <div className="text-xs text-muted-foreground">{gpu.vram}GB VRAM</div>
                            <div className="text-[11px] text-muted-foreground">Base rate</div>
                          </div>
                          <div className="tabular-nums text-sm text-primary">{formatHourly(gpu.baseRate)}</div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Quick Presets */}
              <div>
                <label className="mb-3 block text-sm font-medium text-muted-foreground">{presetsLabel}</label>
                <div className="grid grid-cols-3 gap-2">
                  {presets.map((preset) => (
                    <button
                      key={preset.label}
                      onClick={() => applyPreset(preset.hours, preset.utilization)}
                      className="rounded-md border bg-background/60 px-3 py-2 text-xs transition-colors hover:bg-background"
                    >
                      <div className="font-medium">{preset.label}</div>
                      <div className="text-muted-foreground">
                        {preset.hours}h • {preset.utilization}%
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Hours Per Day */}
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <label className="text-sm font-medium text-muted-foreground">{hoursLabel}</label>
                  <span className="tabular-nums text-lg font-bold text-primary">{hoursPerDay[0]}h</span>
                </div>
                <Slider
                  value={hoursPerDay}
                  onValueChange={setHoursPerDay}
                  min={1}
                  max={24}
                  step={1}
                  aria-label="Hours available per day"
                  className="[&_[role=slider]]:bg-primary"
                />
                <div className="mt-2 flex justify-between text-xs text-muted-foreground">
                  <span>1h</span>
                  <span>24h</span>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">≈ {hoursPerDay[0] * 30}h / mo</div>
              </div>

              {/* Utilization */}
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <label className="text-sm font-medium text-muted-foreground">{utilizationLabel}</label>
                  <span className="tabular-nums text-lg font-bold text-primary">{utilization[0]}%</span>
                </div>
                <Slider
                  value={utilization}
                  onValueChange={setUtilization}
                  min={10}
                  max={100}
                  step={5}
                  aria-label="Expected utilization"
                  className="[&_[role=slider]]:bg-primary"
                />
                <div className="mt-2 flex justify-between text-xs text-muted-foreground">
                  <span>10%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Earnings Display */}
          <div className="animate-in fade-in slide-in-from-bottom-2 rounded-2xl border bg-gradient-to-b from-card to-background p-6 motion-reduce:animate-none sm:p-8">
            <h3 className="mb-6 text-xl font-bold text-foreground">{earningsTitle}</h3>

            <div className="space-y-6">
              {/* Top KPI Card */}
              <div className="rounded-xl border border-primary/20 bg-primary/10 p-6">
                <div className="mb-2 text-sm text-primary">{monthlyLabel}</div>
                <div aria-live="polite" className="tabular-nums text-5xl font-extrabold text-foreground lg:text-6xl">
                  {formatCurrency(monthlyEarnings)}
                </div>
                <div className="mt-2 text-sm text-muted-foreground">{basedOnText(hoursPerDay[0], utilization[0])}</div>
                <div className="mt-3 text-sm font-medium text-emerald-400">
                  {takeHomeLabel}: {formatCurrency(takeHome)}
                </div>
              </div>

              {/* Secondary KPIs */}
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border bg-background/50 p-4">
                  <div className="mb-1 text-sm text-muted-foreground">{dailyLabel}</div>
                  <div className="tabular-nums text-2xl font-bold text-foreground">
                    {formatCurrency(dailyEarnings, { maximumFractionDigits: 2 })}
                  </div>
                </div>
                <div className="rounded-lg border bg-background/50 p-4">
                  <div className="mb-1 text-sm text-muted-foreground">{yearlyLabel}</div>
                  <div className="tabular-nums text-2xl font-bold text-foreground">
                    {formatCurrency(yearlyEarnings)}
                  </div>
                </div>
              </div>

              {/* Breakdown Box */}
              <div className="space-y-3 rounded-lg border bg-background/50 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {breakdownTitle}
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-xs uppercase tracking-wide text-muted-foreground">{hourlyRateLabel}:</span>
                  <span className="tabular-nums text-sm text-foreground">{formatHourly(hourlyRate)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-xs uppercase tracking-wide text-muted-foreground">{hoursPerMonthLabel}:</span>
                  <span className="tabular-nums text-sm text-foreground">{hoursPerDay[0] * 30}h</span>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-xs uppercase tracking-wide text-muted-foreground">
                      {utilizationBreakdownLabel}:
                    </span>
                    <span className="tabular-nums text-sm text-foreground">{utilization[0]}%</span>
                  </div>
                  <div className="mt-2 h-1.5 overflow-hidden rounded bg-primary/15">
                    <div className="h-full rounded bg-primary transition-all" style={{ width: `${utilization[0]}%` }} />
                  </div>
                </div>
                <div className="border-t border-border pt-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-xs uppercase tracking-wide text-muted-foreground">{commissionLabel}:</span>
                    <span className="tabular-nums text-sm text-foreground">
                      -{formatCurrency(commission * monthlyEarnings)}
                    </span>
                  </div>
                  <div className="mt-2 flex justify-between font-medium">
                    <span className="text-foreground">{yourTakeHomeLabel}:</span>
                    <span className="tabular-nums text-primary">{formatCurrency(takeHome)}</span>
                  </div>
                </div>
              </div>

              {/* CTA */}
              <Button
                className="w-full bg-primary text-primary-foreground transition-transform hover:bg-primary/90 active:scale-[0.98]"
                aria-label={ctaAriaLabel}
              >
                {ctaLabel}
              </Button>

              {/* Secondary Link */}
              <button
                onClick={() => {
                  if (gpuListRef.current) {
                    window.scrollTo({
                      top: gpuListRef.current.offsetTop - 80,
                      behavior: 'smooth',
                    })
                  }
                }}
                className="w-full text-center text-sm text-primary underline-offset-4 hover:underline"
              >
                {secondaryCTALabel}
              </button>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 rounded-lg border bg-card/50 p-6 text-center">
          <p className="text-sm text-muted-foreground">{disclaimerText}</p>
        </div>
      </div>
    </div>
  )
}
