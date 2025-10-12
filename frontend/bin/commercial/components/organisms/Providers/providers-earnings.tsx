"use client"

import { useState } from "react"
import { Button } from '@/components/atoms/Button/Button'
import { Slider } from '@/components/atoms/Slider/Slider'
import { cn } from '@/lib/utils'

const gpuModels = [
  { name: "RTX 4090", baseRate: 0.45, vram: 24 },
  { name: "RTX 4080", baseRate: 0.35, vram: 16 },
  { name: "RTX 4070 Ti", baseRate: 0.28, vram: 12 },
  { name: "RTX 3090", baseRate: 0.32, vram: 24 },
  { name: "RTX 3080", baseRate: 0.25, vram: 10 },
  { name: "RTX 3070", baseRate: 0.18, vram: 8 },
]

export function ProvidersEarnings() {
  const [selectedGPU, setSelectedGPU] = useState(gpuModels[0])
  const [utilization, setUtilization] = useState([80])
  const [hoursPerDay, setHoursPerDay] = useState([20])

  const hourlyRate = selectedGPU.baseRate
  const dailyEarnings = hourlyRate * hoursPerDay[0] * (utilization[0] / 100)
  const monthlyEarnings = dailyEarnings * 30
  const yearlyEarnings = monthlyEarnings * 12

  return (
    <section id="earnings-calculator" className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            Calculate Your Potential Earnings
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            See how much you could earn based on your GPU model, availability, and utilization.
          </p>
        </div>

        <div className="mx-auto max-w-4xl">
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Calculator Inputs */}
            <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
              <h3 className="mb-6 text-xl font-bold text-foreground">Your Configuration</h3>

              <div className="space-y-6">
                {/* GPU Selection */}
                <div>
                  <label className="mb-3 block text-sm font-medium text-muted-foreground">Select Your GPU</label>
                  <div className="grid gap-2">
                    {gpuModels.map((gpu) => {
                      const isSelected = selectedGPU.name === gpu.name
                      return (
                        <button
                        key={gpu.name}
                        onClick={() => setSelectedGPU(gpu)}
                        className={cn(
                          'rounded-lg border p-3 text-left transition-all',
                          isSelected
                            ? 'border-primary bg-primary/10'
                            : 'border-border bg-background/50 hover:border-border/70'
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-foreground">{gpu.name}</div>
                            <div className="text-xs text-muted-foreground">{gpu.vram}GB VRAM</div>
                          </div>
                          <div className="text-sm text-primary">€{gpu.baseRate}/hr</div>
                        </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Hours Per Day */}
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <label className="text-sm font-medium text-muted-foreground">Hours Available Per Day</label>
                    <span className="text-lg font-bold text-primary">{hoursPerDay[0]}h</span>
                  </div>
                  <Slider
                    value={hoursPerDay}
                    onValueChange={setHoursPerDay}
                    min={1}
                    max={24}
                    step={1}
                    className="[&_[role=slider]]:bg-primary"
                  />
                  <div className="mt-2 flex justify-between text-xs text-muted-foreground">
                    <span>1h</span>
                    <span>24h</span>
                  </div>
                </div>

                {/* Utilization */}
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <label className="text-sm font-medium text-muted-foreground">Expected Utilization</label>
                    <span className="text-lg font-bold text-primary">{utilization[0]}%</span>
                  </div>
                  <Slider
                    value={utilization}
                    onValueChange={setUtilization}
                    min={10}
                    max={100}
                    step={5}
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
            <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
              <h3 className="mb-6 text-xl font-bold text-foreground">Your Potential Earnings</h3>

              <div className="space-y-6">
                <div className="rounded-xl border border-primary/20 bg-primary/10 p-6">
                  <div className="mb-2 text-sm text-primary">Monthly Earnings</div>
                  <div className="text-5xl font-bold text-foreground">€{monthlyEarnings.toFixed(0)}</div>
                  <div className="mt-2 text-sm text-muted-foreground">
                    Based on {hoursPerDay[0]}h/day at {utilization[0]}% utilization
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-border bg-background/50 p-4">
                    <div className="mb-1 text-sm text-muted-foreground">Daily</div>
                    <div className="text-2xl font-bold text-foreground">€{dailyEarnings.toFixed(2)}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-background/50 p-4">
                    <div className="mb-1 text-sm text-muted-foreground">Yearly</div>
                    <div className="text-2xl font-bold text-foreground">€{yearlyEarnings.toFixed(0)}</div>
                  </div>
                </div>

                <div className="space-y-3 rounded-lg border border-border bg-background/50 p-4">
                  <div className="text-sm font-medium text-muted-foreground">Breakdown</div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Hourly rate:</span>
                    <span className="text-foreground">€{hourlyRate.toFixed(2)}/hr</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Hours per month:</span>
                    <span className="text-foreground">{hoursPerDay[0] * 30}h</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Utilization:</span>
                    <span className="text-foreground">{utilization[0]}%</span>
                  </div>
                  <div className="border-t border-border pt-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">rbee commission (15%):</span>
                      <span className="text-foreground">-€{(monthlyEarnings * 0.15).toFixed(0)}</span>
                    </div>
                    <div className="mt-2 flex justify-between font-medium">
                      <span className="text-foreground">Your take-home:</span>
                      <span className="text-primary">€{(monthlyEarnings * 0.85).toFixed(0)}</span>
                    </div>
                  </div>
                </div>

                <Button className="w-full bg-primary text-primary-foreground hover:bg-primary/90">Start Earning Now</Button>
              </div>
            </div>
          </div>

          <div className="mt-8 rounded-lg border border-border bg-card/50 p-6 text-center">
            <p className="text-sm text-muted-foreground">
              Earnings are estimates based on current market rates and may vary. Actual earnings depend on demand, your
              pricing, and availability.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
