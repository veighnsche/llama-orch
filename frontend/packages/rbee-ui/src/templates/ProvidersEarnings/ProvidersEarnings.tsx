"use client";

import { Button } from "@rbee/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms/Card";
import { EarningsBreakdownCard } from "@rbee/ui/molecules/EarningsBreakdownCard";
import { GPUSelector } from "@rbee/ui/molecules/GPUSelector";
import { IconCardHeader } from "@rbee/ui/molecules/IconCardHeader";
import { LabeledSlider } from "@rbee/ui/molecules/LabeledSlider";
import { OptionSelector } from "@rbee/ui/molecules/OptionSelector";
import { cn } from "@rbee/ui/utils";
import { Cpu, TrendingUp } from "lucide-react";
import { useRef, useState } from "react";

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersEarningsGPUModel = {
  name: string;
  baseRate: number;
  vram: number;
};

export type ProvidersEarningsPreset = {
  label: string;
  hours: number;
  utilization: number;
};

export type ProvidersEarningsProps = {
  gpuModels: ProvidersEarningsGPUModel[];
  presets: ProvidersEarningsPreset[];
  commission: number;
  configTitle: string;
  selectGPULabel: string;
  presetsLabel: string;
  hoursLabel: string;
  utilizationLabel: string;
  earningsTitle: string;
  monthlyLabel: string;
  basedOnText: (hours: number, utilization: number) => string;
  takeHomeLabel: string;
  dailyLabel: string;
  yearlyLabel: string;
  breakdownTitle: string;
  hourlyRateLabel: string;
  hoursPerMonthLabel: string;
  utilizationBreakdownLabel: string;
  commissionLabel: string;
  yourTakeHomeLabel: string;
  ctaLabel: string;
  ctaAriaLabel: string;
  secondaryCTALabel: string;
  formatCurrency: (n: number, opts?: Intl.NumberFormatOptions) => string;
  formatHourly: (n: number) => string;
};

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersEarnings - Interactive earnings calculator for GPU providers
 *
 * @example
 * ```tsx
 * <ProvidersEarnings
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
export function ProvidersEarnings({
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
  formatCurrency,
  formatHourly,
}: ProvidersEarningsProps) {
  const [selectedGPU, setSelectedGPU] = useState(gpuModels[0]);
  const [utilization, setUtilization] = useState([80]);
  const [hoursPerDay, setHoursPerDay] = useState([20]);
  const gpuListRef = useRef<HTMLDivElement>(null);

  const applyPreset = (hours: number, util: number) => {
    setHoursPerDay([hours]);
    setUtilization([util]);
  };

  const hourlyRate = selectedGPU.baseRate;
  const dailyEarnings = hourlyRate * hoursPerDay[0] * (utilization[0] / 100);
  const monthlyEarnings = dailyEarnings * 30;
  const yearlyEarnings = monthlyEarnings * 12;
  const takeHome = monthlyEarnings * (1 - commission);

  return (
    <div id="earnings-calculator">
      <div className="mx-auto max-w-4xl">
        <div className="grid gap-8 lg:grid-cols-2 lg:items-start">
          {/* Calculator Inputs */}
          <Card className="animate-in fade-in slide-in-from-bottom-2 bg-gradient-to-b from-card to-background shadow-lg motion-reduce:animate-none">
            <IconCardHeader
              icon={<Cpu />}
              title={configTitle}
              iconSize="md"
              iconTone="primary"
              titleClassName="text-xl"
            />

            <CardContent className="space-y-6">
              {/* GPU Selection */}
              <GPUSelector
                models={gpuModels}
                selectedModel={selectedGPU}
                onSelect={setSelectedGPU}
                label={selectGPULabel}
                formatHourly={formatHourly}
                containerRef={gpuListRef}
              />

              {/* Quick Presets */}
              <div>
                <label className="mb-3 block text-sm font-medium text-muted-foreground">
                  {presetsLabel}
                </label>
                <OptionSelector
                  options={presets.map((preset) => ({
                  id: preset.label.toLowerCase().replace(/\s+/g, "-"),
                  label: preset.label,
                  subtitle: `${preset.hours}h • ${preset.utilization}%`,
                  data: {
                    hours: preset.hours,
                    utilization: preset.utilization,
                  },
                }))}
                onSelect={(data) => applyPreset(data.hours, data.utilization)}
                columns={3}
                />
              </div>

              {/* Hours Per Day */}
              <LabeledSlider
                label={hoursLabel}
                value={hoursPerDay}
                onValueChange={setHoursPerDay}
                min={1}
                max={24}
                step={1}
                ariaLabel="Hours available per day"
                formatValue={(v) => `${v}h`}
                minLabel="1h"
                maxLabel="24h"
                helperText={(v) => `≈ ${v * 30}h / mo`}
              />

              {/* Utilization */}
              <LabeledSlider
                label={utilizationLabel}
                value={utilization}
                onValueChange={setUtilization}
                min={10}
                max={100}
                step={5}
                ariaLabel="Expected utilization"
                formatValue={(v) => `${v}%`}
                minLabel="10%"
                maxLabel="100%"
              />
            </CardContent>
          </Card>

          {/* Earnings Display */}
          <Card className="animate-in fade-in slide-in-from-bottom-2 [animation-delay:100ms] bg-gradient-to-b from-card to-background shadow-lg motion-reduce:animate-none lg:sticky lg:top-24">
            <IconCardHeader
              icon={<TrendingUp />}
              title={earningsTitle}
              iconSize="md"
              iconTone="primary"
              titleClassName="text-xl"
            />

            <CardContent className="space-y-6">
              {/* Top KPI Card */}
              <Card className="border-primary/20 bg-primary/10 shadow-sm">
                <CardHeader className="pb-0">
                  <CardTitle className="text-sm text-primary">
                    {monthlyLabel}{" "}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div
                    aria-live="polite"
                    className="tabular-nums text-5xl font-extrabold text-foreground lg:text-6xl"
                  >
                    {formatCurrency(monthlyEarnings)}
                  </div>
                  <div className="mt-2 text-sm text-muted-foreground">
                    {basedOnText(hoursPerDay[0], utilization[0])}
                  </div>
                  <div className="mt-3 text-sm font-medium text-emerald-400">
                    {takeHomeLabel}: {formatCurrency(takeHome)}
                  </div>
                </CardContent>
              </Card>

              {/* Secondary KPIs */}
              <div className="grid gap-4 sm:grid-cols-2">
                <Card className="bg-background/50 shadow-sm transition-shadow hover:shadow-md">
                  <CardHeader className="pb-0">
                    <CardTitle className="text-sm text-muted-foreground">
                      {dailyLabel}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="tabular-nums text-2xl font-bold text-foreground">
                      {formatCurrency(dailyEarnings, {
                        maximumFractionDigits: 2,
                      })}
                    </div>
                  </CardContent>
                </Card>
                <Card className="bg-background/50 shadow-sm transition-shadow hover:shadow-md">
                  <CardHeader className="pb-0">
                    <CardTitle className="text-sm text-muted-foreground">
                      {yearlyLabel}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="tabular-nums text-2xl font-bold text-foreground">
                      {formatCurrency(yearlyEarnings)}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Breakdown Box */}
              <EarningsBreakdownCard
                title={breakdownTitle}
                hourlyRate={{
                  label: hourlyRateLabel,
                  value: formatHourly(hourlyRate),
                }}
                hoursPerMonth={{
                  label: hoursPerMonthLabel,
                  value: `${hoursPerDay[0] * 30}h`,
                }}
                utilization={{
                  label: utilizationBreakdownLabel,
                  value: utilization[0],
                }}
                commission={{
                  label: commissionLabel,
                  value: `-${formatCurrency(commission * monthlyEarnings)}`,
                }}
                takeHome={{
                  label: yourTakeHomeLabel,
                  value: formatCurrency(takeHome),
                }}
              />

              {/* CTA */}
              <Button
                className="w-full bg-primary text-primary-foreground shadow-md transition-all hover:bg-primary/90 hover:shadow-lg active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
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
                      behavior: "smooth",
                    });
                  }
                }}
                className="w-full text-center text-sm font-medium text-primary underline-offset-4 transition-colors hover:underline hover:text-primary/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 rounded-md py-2"
              >
                {secondaryCTALabel}
              </button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
