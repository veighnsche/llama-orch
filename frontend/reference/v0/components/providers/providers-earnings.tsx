"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"

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
    <section id="earnings-calculator" className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">
            Calculate Your Potential Earnings
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            See how much you could earn based on your GPU model, availability, and utilization.
          </p>
        </div>

        <div className="mx-auto max-w-4xl">
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Calculator Inputs */}
            <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
              <h3 className="mb-6 text-xl font-bold text-white">Your Configuration</h3>

              <div className="space-y-6">
                {/* GPU Selection */}
                <div>
                  <label className="mb-3 block text-sm font-medium text-slate-300">Select Your GPU</label>
                  <div className="grid gap-2">
                    {gpuModels.map((gpu) => (
                      <button
                        key={gpu.name}
                        onClick={() => setSelectedGPU(gpu)}
                        className={`rounded-lg border p-3 text-left transition-all ${
                          selectedGPU.name === gpu.name
                            ? "border-amber-500 bg-amber-500/10"
                            : "border-slate-800 bg-slate-950/50 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-white">{gpu.name}</div>
                            <div className="text-xs text-slate-400">{gpu.vram}GB VRAM</div>
                          </div>
                          <div className="text-sm text-amber-400">€{gpu.baseRate}/hr</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Hours Per Day */}
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-300">Hours Available Per Day</label>
                    <span className="text-lg font-bold text-amber-400">{hoursPerDay[0]}h</span>
                  </div>
                  <Slider
                    value={hoursPerDay}
                    onValueChange={setHoursPerDay}
                    min={1}
                    max={24}
                    step={1}
                    className="[&_[role=slider]]:bg-amber-500"
                  />
                  <div className="mt-2 flex justify-between text-xs text-slate-400">
                    <span>1h</span>
                    <span>24h</span>
                  </div>
                </div>

                {/* Utilization */}
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-300">Expected Utilization</label>
                    <span className="text-lg font-bold text-amber-400">{utilization[0]}%</span>
                  </div>
                  <Slider
                    value={utilization}
                    onValueChange={setUtilization}
                    min={10}
                    max={100}
                    step={5}
                    className="[&_[role=slider]]:bg-amber-500"
                  />
                  <div className="mt-2 flex justify-between text-xs text-slate-400">
                    <span>10%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Earnings Display */}
            <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
              <h3 className="mb-6 text-xl font-bold text-white">Your Potential Earnings</h3>

              <div className="space-y-6">
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 p-6">
                  <div className="mb-2 text-sm text-amber-400">Monthly Earnings</div>
                  <div className="text-5xl font-bold text-white">€{monthlyEarnings.toFixed(0)}</div>
                  <div className="mt-2 text-sm text-slate-400">
                    Based on {hoursPerDay[0]}h/day at {utilization[0]}% utilization
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                    <div className="mb-1 text-sm text-slate-400">Daily</div>
                    <div className="text-2xl font-bold text-white">€{dailyEarnings.toFixed(2)}</div>
                  </div>
                  <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                    <div className="mb-1 text-sm text-slate-400">Yearly</div>
                    <div className="text-2xl font-bold text-white">€{yearlyEarnings.toFixed(0)}</div>
                  </div>
                </div>

                <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-950/50 p-4">
                  <div className="text-sm font-medium text-slate-300">Breakdown</div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Hourly rate:</span>
                    <span className="text-white">€{hourlyRate.toFixed(2)}/hr</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Hours per month:</span>
                    <span className="text-white">{hoursPerDay[0] * 30}h</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Utilization:</span>
                    <span className="text-white">{utilization[0]}%</span>
                  </div>
                  <div className="border-t border-slate-800 pt-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">rbee commission (15%):</span>
                      <span className="text-white">-€{(monthlyEarnings * 0.15).toFixed(0)}</span>
                    </div>
                    <div className="mt-2 flex justify-between font-medium">
                      <span className="text-white">Your take-home:</span>
                      <span className="text-amber-400">€{(monthlyEarnings * 0.85).toFixed(0)}</span>
                    </div>
                  </div>
                </div>

                <Button className="w-full bg-amber-500 text-slate-950 hover:bg-amber-400">Start Earning Now</Button>
              </div>
            </div>
          </div>

          <div className="mt-8 rounded-lg border border-slate-800 bg-slate-900/50 p-6 text-center">
            <p className="text-sm text-slate-400">
              Earnings are estimates based on current market rates and may vary. Actual earnings depend on demand, your
              pricing, and availability.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
