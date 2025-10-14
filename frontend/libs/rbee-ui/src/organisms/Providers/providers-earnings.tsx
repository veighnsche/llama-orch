'use client'

import { useState, useRef } from 'react'
import { Cpu } from 'lucide-react'
import { Button } from '@rbee/ui/atoms/Button'
import { Slider } from '@rbee/ui/atoms/Slider'
import { cn } from '@rbee/ui/utils'

const COMMISSION = 0.15

const fmt = (n: number, opts: Intl.NumberFormatOptions = {}) =>
  new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0, ...opts }).format(n)

const fmtHr = (n: number) => `${fmt(n, { maximumFractionDigits: 2 })}/hr`

const gpuModels = [
  { name: 'RTX 4090', baseRate: 0.45, vram: 24 },
  { name: 'RTX 4080', baseRate: 0.35, vram: 16 },
  { name: 'RTX 4070 Ti', baseRate: 0.28, vram: 12 },
  { name: 'RTX 3090', baseRate: 0.32, vram: 24 },
  { name: 'RTX 3080', baseRate: 0.25, vram: 10 },
  { name: 'RTX 3070', baseRate: 0.18, vram: 8 },
]

const presets = [
  { label: 'Casual', hours: 8, utilization: 50 },
  { label: 'Daily', hours: 16, utilization: 70 },
  { label: 'Always On', hours: 24, utilization: 90 },
]

export function ProvidersEarnings() {
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
  const takeHome = monthlyEarnings * (1 - COMMISSION)

  return (
    <section
      id="earnings-calculator"
      className="border-b border-border bg-gradient-to-b from-background via-primary/5 to-background px-6 py-20 lg:py-28"
    >
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <div className="mb-2 text-sm font-medium text-primary/80">Estimate Your Earnings</div>
          <h2 className="mb-4 text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">
            Calculate Your Potential Earnings
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-lg text-muted-foreground">
            See what you could earn based on GPU model, availability, and utilization.
          </p>
        </div>

        <div className="mx-auto max-w-4xl">
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Calculator Inputs */}
            <div className="animate-in fade-in slide-in-from-bottom-2 rounded-2xl border border-border bg-gradient-to-b from-card to-background p-6 motion-reduce:animate-none sm:p-8">
              <div className="mb-6 flex items-center gap-2">
                <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />
                <h3 className="text-xl font-bold text-foreground">Your Configuration</h3>
              </div>

              <div className="space-y-6">
                {/* GPU Selection */}
                <div ref={gpuListRef}>
                  <label className="mb-3 block text-sm font-medium text-muted-foreground">Select Your GPU</label>
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
                            <div className="tabular-nums text-sm text-primary">{fmtHr(gpu.baseRate)}</div>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Quick Presets */}
                <div>
                  <label className="mb-3 block text-sm font-medium text-muted-foreground">Quick Presets</label>
                  <div className="grid grid-cols-3 gap-2">
                    {presets.map((preset) => (
                      <button
                        key={preset.label}
                        onClick={() => applyPreset(preset.hours, preset.utilization)}
                        className="rounded-md border border-border bg-background/60 px-3 py-2 text-xs transition-colors hover:bg-background"
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
                    <label className="text-sm font-medium text-muted-foreground">Hours Available Per Day</label>
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
                    <label className="text-sm font-medium text-muted-foreground">Expected Utilization</label>
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
            <div className="animate-in fade-in slide-in-from-bottom-2 rounded-2xl border border-border bg-gradient-to-b from-card to-background p-6 motion-reduce:animate-none sm:p-8">
              <h3 className="mb-6 text-xl font-bold text-foreground">Your Potential Earnings</h3>

              <div className="space-y-6">
                {/* Top KPI Card */}
                <div className="rounded-xl border border-primary/20 bg-primary/10 p-6">
                  <div className="mb-2 text-sm text-primary">Monthly Earnings</div>
                  <div aria-live="polite" className="tabular-nums text-5xl font-extrabold text-foreground lg:text-6xl">
                    {fmt(monthlyEarnings)}
                  </div>
                  <div className="mt-2 text-sm text-muted-foreground">
                    Based on {hoursPerDay[0]}h/day at {utilization[0]}% utilization
                  </div>
                  <div className="mt-3 text-sm font-medium text-emerald-400">
                    Take-home (after 15%): {fmt(takeHome)}
                  </div>
                </div>

                {/* Secondary KPIs */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-border bg-background/50 p-4">
                    <div className="mb-1 text-sm text-muted-foreground">Daily</div>
                    <div className="tabular-nums text-2xl font-bold text-foreground">
                      {fmt(dailyEarnings, { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-background/50 p-4">
                    <div className="mb-1 text-sm text-muted-foreground">Yearly</div>
                    <div className="tabular-nums text-2xl font-bold text-foreground">{fmt(yearlyEarnings)}</div>
                  </div>
                </div>

                {/* Breakdown Box */}
                <div className="space-y-3 rounded-lg border border-border bg-background/50 p-4">
                  <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Breakdown</div>
                  <div className="flex justify-between text-sm">
                    <span className="text-xs uppercase tracking-wide text-muted-foreground">Hourly rate:</span>
                    <span className="tabular-nums text-sm text-foreground">{fmtHr(hourlyRate)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-xs uppercase tracking-wide text-muted-foreground">Hours per month:</span>
                    <span className="tabular-nums text-sm text-foreground">{hoursPerDay[0] * 30}h</span>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm">
                      <span className="text-xs uppercase tracking-wide text-muted-foreground">Utilization:</span>
                      <span className="tabular-nums text-sm text-foreground">{utilization[0]}%</span>
                    </div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded bg-primary/15">
                      <div
                        className="h-full rounded bg-primary transition-all"
                        style={{ width: `${utilization[0]}%` }}
                      />
                    </div>
                  </div>
                  <div className="border-t border-border pt-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-xs uppercase tracking-wide text-muted-foreground">
                        rbee commission (15%):
                      </span>
                      <span className="tabular-nums text-sm text-foreground">-{fmt(COMMISSION * monthlyEarnings)}</span>
                    </div>
                    <div className="mt-2 flex justify-between font-medium">
                      <span className="text-foreground">Your take-home:</span>
                      <span className="tabular-nums text-primary">{fmt(takeHome)}</span>
                    </div>
                  </div>
                </div>

                {/* CTA */}
                <Button
                  className="w-full bg-primary text-primary-foreground transition-transform hover:bg-primary/90 active:scale-[0.98]"
                  aria-label="Start earning with rbee"
                >
                  Start Earning Now
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
                  Estimate on another GPU
                </button>
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="mt-8 rounded-lg border border-border bg-card/50 p-6 text-center">
            <p className="text-sm text-muted-foreground">
              Earnings are estimates based on current market rates and may vary. Actual earnings depend on demand, your
              pricing, and availability. Figures are estimates.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
