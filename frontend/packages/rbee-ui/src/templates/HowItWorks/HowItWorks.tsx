import { CodeBlock, StepNumber, TerminalWindow } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export type StepBlock =
  | { kind: 'terminal'; title?: string; lines: ReactNode; copyText?: string }
  | { kind: 'code'; title?: string; language?: string; code: string }
  | { kind: 'note'; content: ReactNode }

export type HowItWorksProps = {
  steps: Array<{
    label: string
    number?: number
    block?: StepBlock
  }>
  className?: string
}

export function HowItWorks({ steps, className }: HowItWorksProps) {
  return (
    <div className={className}>
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
              <StepNumber number={stepNumber} size="md" variant="primary" className="flex-shrink-0 rounded-lg" />

              {/* Step content */}
              <div className="flex-1">
                <h3 className="mb-3 text-xl font-semibold text-card-foreground">{step.label}</h3>

                {/* Render block */}
                {step.block && (
                  <>
                    {step.block.kind === 'terminal' && (
                      <TerminalWindow
                        showChrome
                        title={step.block.title || 'terminal'}
                        copyable
                        copyText={step.block.copyText}
                      >
                        {step.block.lines}
                      </TerminalWindow>
                    )}

                    {step.block.kind === 'code' && (
                      <CodeBlock
                        code={step.block.code}
                        language={step.block.language}
                        title={step.block.title}
                        copyable
                      />
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
    </div>
  )
}
