'use client'

import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader, StatusKPI, TerminalWindow, TimelineStep } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface MetricKPI {
  icon: ReactNode
  color: 'chart-3' | 'primary' | 'chart-2'
  label: string
  value: string
  progressPercentage: number
}

export interface CancellationStep {
  timestamp: string
  title: ReactNode
  description: string
  variant?: 'success'
}

export interface RealTimeProgressTemplateProps {
  narrationTitle: string
  narrationSubtitle: string
  terminalTitle: string
  terminalAriaLabel: string
  terminalContent: ReactNode
  terminalFooter?: ReactNode
  metricKPIs: MetricKPI[]
  cancellationTitle: string
  cancellationSubtitle: string
  cancellationSteps: CancellationStep[]
  className?: string
}

export function RealTimeProgressTemplate({
  narrationTitle,
  narrationSubtitle,
  terminalTitle,
  terminalAriaLabel,
  terminalContent,
  terminalFooter,
  metricKPIs,
  cancellationTitle,
  cancellationSubtitle,
  cancellationSteps,
  className,
}: RealTimeProgressTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl space-y-10">
        <div>
          <IconCardHeader
            icon={metricKPIs[0].icon}
            iconTone="primary"
            iconSize="md"
            title={narrationTitle}
            subtitle={narrationSubtitle}
            useCardHeader={false}
            className="mb-4"
          />

          <TerminalWindow
            title={terminalTitle}
            ariaLabel={terminalAriaLabel}
            className="animate-in fade-in slide-in-from-bottom-2"
            footer={terminalFooter}
          >
            {terminalContent}
          </TerminalWindow>
        </div>

        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-100">
          {metricKPIs.map((kpi, idx) => (
            <div key={idx} className="hover:-translate-y-0.5 transition-transform">
              <StatusKPI icon={kpi.icon} color={kpi.color} label={kpi.label} value={kpi.value} />
              <div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
                <div className={`h-full bg-${kpi.color}`} style={{ width: `${kpi.progressPercentage}%` }} />
              </div>
            </div>
          ))}
        </div>

        <Card className="animate-in fade-in slide-in-from-bottom-2 delay-150">
          <CardContent className="p-6">
            <IconCardHeader
              icon={metricKPIs[0].icon}
              iconTone="warning"
              iconSize="md"
              title={cancellationTitle}
              subtitle={cancellationSubtitle}
              useCardHeader={false}
              className="mb-4"
            />

            <ol className="grid gap-3 sm:grid-cols-4 text-sm" aria-label="Cancellation sequence">
              {cancellationSteps.map((step, idx) => (
                <TimelineStep
                  key={idx}
                  timestamp={step.timestamp}
                  title={step.title}
                  description={step.description}
                  variant={step.variant}
                />
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
