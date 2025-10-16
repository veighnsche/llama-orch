'use client'

import { Badge, Card, CardContent, Separator } from '@rbee/ui/atoms'
import { IconCardHeader, IconPlate, TerminalWindow } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'
import { ArrowDown, CheckCircle2, GitBranch, Network } from 'lucide-react'
import type { ReactNode } from 'react'

export interface DiagramNode {
  name: string
  label: string
  tone: 'primary' | 'chart-2' | 'chart-3'
}

export interface LegendItem {
  label: string
}

export interface BenefitCard {
  icon: LucideIcon
  title: string
  description: string
}

export interface CrossNodeOrchestrationTemplateProps {
  /** Terminal content for pool registry */
  terminalContent: ReactNode
  /** Terminal copyable text */
  terminalCopyText?: string
  /** Benefit cards (3 items) */
  benefits: BenefitCard[]
  /** Diagram nodes for provisioning flow */
  diagramNodes: DiagramNode[]
  /** Diagram arrows with labels */
  diagramArrows: Array<{ label: string; indent?: string }>
  /** Legend items */
  legendItems: LegendItem[]
  /** Provisioning title */
  provisioningTitle: string
  /** Provisioning subtitle */
  provisioningSubtitle: string
  /** Custom class name */
  className?: string
}

/**
 * CrossNodeOrchestrationTemplate - Template for cross-node orchestration features
 *
 * @example
 * ```tsx
 * <CrossNodeOrchestrationTemplate
 *   title="Cross-Pool Orchestration"
 *   subtitle="Seamlessly orchestrate AI workloads across your entire network."
 *   benefits={[...]}
 *   diagramNodes={[...]}
 * />
 * ```
 */
export function CrossNodeOrchestrationTemplate({
  terminalContent,
  terminalCopyText,
  benefits,
  diagramNodes,
  diagramArrows,
  legendItems,
  provisioningTitle,
  provisioningSubtitle,
  className,
}: CrossNodeOrchestrationTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          {/* Pool Registry Management Card */}
          <Card className="animate-in fade-in slide-in-from-left-4 duration-500">
            <IconCardHeader
              icon={<Network className="w-6 h-6" />}
              title="Pool Registry Management"
              subtitle="Configure remote machines once. rbee-keeper handles SSH, validates connectivity, and keeps your pool registry synced."
              iconTone="primary"
              iconSize="md"
            />
            <CardContent className="space-y-6">
              <TerminalWindow
                showChrome={false}
                variant="terminal"
                copyable
                copyText={terminalCopyText}
                className="font-mono text-sm"
              >
                {terminalContent}
              </TerminalWindow>

              <div className="grid sm:grid-cols-3 gap-3">
                {benefits.map((benefit, idx) => (
                  <div
                    key={idx}
                    className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform"
                  >
                    <benefit.icon className="size-5 shrink-0 mt-0.5 text-chart-3" aria-hidden="true" />
                    <div>
                      <div className="font-semibold text-foreground text-sm">{benefit.title}</div>
                      <div className="text-xs text-muted-foreground mt-1">{benefit.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Separator className="lg:hidden my-2 opacity-40" />

          {/* Automatic Worker Provisioning Card */}
          <Card className="animate-in fade-in slide-in-from-right-4 duration-500 delay-100">
            <CardContent className="space-y-6 pt-6">
              <div className="flex items-start gap-4">
                <IconPlate
                  icon={<GitBranch className="w-6 h-6" />}
                  tone="chart-2"
                  size="md"
                  shape="rounded"
                  className="flex-shrink-0"
                />
                <div>
                  <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">{provisioningTitle}</h3>
                  <p className="text-muted-foreground leading-relaxed">{provisioningSubtitle}</p>
                </div>
              </div>

              {/* Diagram */}
              <div className="relative bg-background rounded-xl p-6 overflow-hidden">
                <div className="space-y-4">
                  {/* Row 1: First node */}
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <DiagramNodeComponent
                        name={diagramNodes[0].name}
                        label={diagramNodes[0].label}
                        tone={diagramNodes[0].tone}
                      />
                    </div>
                  </div>

                  {/* Arrow 1 */}
                  {diagramArrows[0] && (
                    <DiagramArrowComponent label={diagramArrows[0].label} indent={diagramArrows[0].indent} />
                  )}

                  {/* Row 2: Second node */}
                  {diagramNodes[1] && (
                    <div className={cn('flex items-center gap-3', diagramArrows[0]?.indent)}>
                      <div className="flex-1">
                        <DiagramNodeComponent
                          name={diagramNodes[1].name}
                          label={diagramNodes[1].label}
                          tone={diagramNodes[1].tone}
                        />
                      </div>
                    </div>
                  )}

                  {/* Arrow 2 */}
                  {diagramArrows[1] && (
                    <DiagramArrowComponent label={diagramArrows[1].label} indent={diagramArrows[1].indent} />
                  )}

                  {/* Row 3: Third node */}
                  {diagramNodes[2] && (
                    <div className={cn('flex items-center gap-3', diagramArrows[1]?.indent)}>
                      <div className="flex-1">
                        <DiagramNodeComponent
                          name={diagramNodes[2].name}
                          label={diagramNodes[2].label}
                          tone={diagramNodes[2].tone}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Legend */}
              <div className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                {legendItems.map((item, idx) => (
                  <LegendItemComponent key={idx} label={item.label} />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

// Helper Components
interface DiagramNodeComponentProps {
  name: string
  label: string
  tone: 'primary' | 'chart-2' | 'chart-3'
}

function DiagramNodeComponent({ name, label, tone }: DiagramNodeComponentProps) {
  const toneClasses = {
    primary: 'bg-primary/10 border-primary/20',
    'chart-2': 'bg-chart-2/10 border-chart-2/20',
    'chart-3': 'bg-chart-3/10 border-chart-3/20',
  }

  return (
    <div className={cn('rounded-lg border-2 p-3 transition-all hover:scale-105', toneClasses[tone])}>
      <div className="font-mono text-sm font-semibold text-foreground">{name}</div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </div>
  )
}

interface DiagramArrowComponentProps {
  label: string
  indent?: string
}

function DiagramArrowComponent({ label, indent }: DiagramArrowComponentProps) {
  return (
    <div className={cn('flex items-center gap-2', indent)}>
      <ArrowDown className="size-4 text-muted-foreground" aria-hidden="true" />
      <span className="text-xs text-muted-foreground font-mono">{label}</span>
    </div>
  )
}

interface LegendItemComponentProps {
  label: string
}

function LegendItemComponent({ label }: LegendItemComponentProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="size-2 rounded-full bg-chart-3 shrink-0" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
