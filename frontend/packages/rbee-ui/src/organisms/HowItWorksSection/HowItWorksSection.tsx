import { ConsoleOutput } from '@rbee/ui/atoms'
import { SectionContainer } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export type StepBlock =
  | { kind: 'terminal'; title?: string; lines: ReactNode; copyText?: string }
  | { kind: 'code'; title?: string; language?: string; lines: ReactNode; copyText?: string }
  | { kind: 'note'; content: ReactNode }

export type HowItWorksSectionProps = {
  title: string
  subtitle?: string
  steps: Array<{
    label: string
    number?: number
    block?: StepBlock
  }>
  id?: string
  className?: string
}

export function HowItWorksSection({ title, subtitle, steps, id, className }: HowItWorksSectionProps) {
  return (
    <SectionContainer
      title={title}
      description={subtitle}
      bgVariant="secondary"
      paddingY="2xl"
      maxWidth="7xl"
      align="center"
      headingId={id}
      className={cn('border-b border-border', className)}
    >
      {/* Steps */}
      <div className="mx-auto max-w-4xl space-y-12">
        {steps.map((step, index) => {
          const stepNumber = step.number ?? index + 1
          return (
            <div
              key={index}
              className={cn(
                'flex gap-6 animate-in slide-in-from-bottom-2 fade-in duration-500',
                'border-t border-border/60 pt-8 sm:border-0 sm:pt-0',
              )}
              style={{ animationDelay: `${index * 120}ms` }}
            >
              {/* Step badge */}
              <div
                className="grid h-12 w-12 flex-shrink-0 place-content-center rounded-lg bg-primary text-xl font-bold text-primary-foreground"
                aria-hidden="true"
              >
                {stepNumber}
              </div>

              {/* Step content */}
              <div className="flex-1">
                <h3 className="mb-3 text-xl font-semibold text-card-foreground">{step.label}</h3>

                {/* Render block */}
                {step.block && (
                  <>
                    {step.block.kind === 'terminal' && (
                      <ConsoleOutput
                        showChrome
                        title={step.block.title || 'terminal'}
                        background="dark"
                        copyable
                        copyText={step.block.copyText}
                      >
                        {step.block.lines}
                      </ConsoleOutput>
                    )}

                    {step.block.kind === 'code' && (
                      <ConsoleOutput
                        showChrome
                        title={step.block.title || step.block.language || 'code'}
                        background="dark"
                        copyable
                        copyText={step.block.copyText}
                      >
                        {step.block.lines}
                      </ConsoleOutput>
                    )}

                    {step.block.kind === 'note' && (
                      <div className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
                        {step.block.content}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </SectionContainer>
  )
}
