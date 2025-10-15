'use client'

import { ConsoleOutput } from '@rbee/ui/atoms'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export type FeatureItem = {
  id: string
  title: string
  description?: string
  icon: ReactNode
  benefit?: { label?: string; text: string; tone?: 'primary' | 'muted' | 'destructive' }
  example?: {
    kind: 'terminal' | 'code' | 'markdown'
    title?: string
    language?: string
    content: string
    copyText?: string
  }
}

export type FeatureTabsSectionProps = {
  title: string
  subtitle?: string
  items: FeatureItem[]
  defaultId?: string
  id?: string
  className?: string
}

export function FeatureTabsSection({ title, subtitle, items, defaultId, id, className }: FeatureTabsSectionProps) {
  const defaultValue = defaultId || items[0]?.id

  const getBenefitToneClasses = (tone?: 'primary' | 'muted' | 'destructive') => {
    switch (tone) {
      case 'primary':
        return 'border-primary/30 bg-primary/10 text-primary'
      case 'destructive':
        return 'border-destructive/30 bg-destructive/10 text-destructive'
      case 'muted':
      default:
        return 'border-border bg-card/60 text-foreground'
    }
  }

  return (
    <section id={id} className={cn('border-b border-border py-24', className)}>
      <div className="container mx-auto px-4">
        <Tabs defaultValue={defaultValue} className="w-full" orientation="horizontal">
          <div className="grid gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
            {/* Left rail: sticky intro + TabsList */}
            <div className="lg:sticky lg:top-24 self-start space-y-6">
              <div>
                <h2 className="text-3xl font-bold tracking-tight text-foreground">{title}</h2>
                {subtitle && <p className="mt-2 text-muted-foreground">{subtitle}</p>}
              </div>

              <TabsList aria-label={title}>
                {items.map((item) => (
                  <TabsTrigger
                    key={item.id}
                    value={item.id}
                    className="flex-col lg:flex-row items-start lg:items-center"
                  >
                    <span className="flex items-center gap-3 w-full">
                      <span
                        className="size-4 text-muted-foreground group-data-[state=active]:text-primary"
                        aria-hidden="true"
                      >
                        {item.icon}
                      </span>
                      <span className="font-semibold">{item.title}</span>
                    </span>
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>

            {/* Right column: content panels */}
            <div className="relative min-h-[600px]">
              {items.map((item) => (
                <TabsContent
                  key={item.id}
                  value={item.id}
                  className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible"
                >
                  <div className="bg-card border rounded-2xl p-8 space-y-6">
                    {/* Header */}
                    <div>
                      <h3 className="text-2xl font-bold text-foreground mb-3">{item.title}</h3>
                      {item.description && <p className="text-muted-foreground leading-relaxed">{item.description}</p>}
                    </div>

                    {/* Example */}
                    {item.example && (
                      <>
                        {(item.example.kind === 'terminal' || item.example.kind === 'code') && (
                          <ConsoleOutput
                            showChrome
                            title={
                              item.example.title ||
                              (item.example.kind === 'terminal' ? 'terminal' : item.example.language || 'code')
                            }
                            background="dark"
                            copyable
                            copyText={item.example.copyText || item.example.content}
                          >
                            <pre className="text-sm">
                              <code>{item.example.content}</code>
                            </pre>
                          </ConsoleOutput>
                        )}
                        {item.example.kind === 'markdown' && (
                          <div className="overflow-hidden rounded-lg border bg-card">
                            {item.example.title && (
                              <div className="border-b border-border bg-muted px-4 py-2">
                                <span className="text-sm text-muted-foreground">{item.example.title}</span>
                              </div>
                            )}
                            <div className="p-4">
                              <div className="prose prose-sm dark:prose-invert max-w-none">{item.example.content}</div>
                            </div>
                          </div>
                        )}
                      </>
                    )}

                    {/* Benefit */}
                    {item.benefit && (
                      <div
                        className={cn(
                          'rounded-lg border p-4 flex items-center gap-2',
                          getBenefitToneClasses(item.benefit.tone),
                        )}
                      >
                        <p className="font-medium">{item.benefit.text}</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              ))}
            </div>
          </div>
        </Tabs>
      </div>
    </section>
  )
}
