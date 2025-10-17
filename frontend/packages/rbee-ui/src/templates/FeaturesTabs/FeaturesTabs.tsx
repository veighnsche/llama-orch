import { Alert, AlertDescription } from '@rbee/ui/atoms/Alert'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { BulletListItem } from '@rbee/ui/molecules/BulletListItem'
import { Check } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn, parseInlineMarkdown } from '@rbee/ui/utils'

export interface TabConfig {
  value: string
  icon: ReactNode
  label: string
  mobileLabel: string
  subtitle: string
  badge: string
  description: string
  content: ReactNode
  highlight: {
    text: string
    variant: 'primary' | 'success' | 'default' | 'info' | 'warning' | 'destructive'
  }
  benefits: Array<{
    text: string
  }>
}

export interface FeaturesTabsProps {
  title: string
  description: string
  tabs: TabConfig[]
  defaultTab?: string
  /** @deprecated No longer used - background handled by TemplateContainer */
  sectionId?: string
  /** @deprecated No longer used - background handled by TemplateContainer */
  bgClassName?: string
}

export function FeaturesTabs({ title, description, tabs, defaultTab }: FeaturesTabsProps) {
  return (
    <div className="container mx-auto px-4">
      <Tabs defaultValue={defaultTab || tabs[0]?.value} className="w-full" orientation="horizontal">
        <div className="grid gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
          {/* Left rail: sticky intro + TabsList */}
          <div className="lg:sticky lg:top-24 self-start space-y-6">
            <div>
              <h2 className="text-3xl font-bold tracking-tight text-foreground">{title}</h2>
              <p className="mt-2 text-muted-foreground">{parseInlineMarkdown(description)}</p>
            </div>

            <TabsList aria-label="Core features">
              {tabs.map((tab) => (
                <TabsTrigger
                  key={tab.value}
                  value={tab.value}
                  className="flex-col lg:flex-row items-start lg:items-center"
                >
                  <span className="flex items-center gap-3 w-full">
                    {tab.icon}
                    <span className="font-semibold">
                      <span className="hidden sm:inline">{tab.label}</span>
                      <span className="sm:hidden">{tab.mobileLabel}</span>
                    </span>
                  </span>
                  <span className="mt-0.5 block text-xs text-muted-foreground text-right font-sans hidden lg:block w-full">
                    {tab.subtitle}
                  </span>
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          {/* Right column: content panels */}
          <div className="relative min-h-[600px]">
            {tabs.map((tab) => (
              <TabsContent
                key={tab.value}
                value={tab.value}
                className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible"
              >
                <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-2xl font-bold text-foreground">{tab.label}</h3>
                      <Badge variant="secondary">{tab.badge}</Badge>
                    </div>
                    <div className="space-y-1">
                      <div className="text-sm font-medium text-foreground">{tab.subtitle}</div>
                      <div className="text-xs text-muted-foreground">{parseInlineMarkdown(tab.description)}</div>
                    </div>
                  </div>

                  {tab.content}

                  <Alert variant={tab.highlight.variant}>
                    <Check />
                    <AlertDescription className="font-medium">{tab.highlight.text}</AlertDescription>
                  </Alert>

                  <div className="pt-2">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Why it matters</h4>
                    <ul className="grid sm:grid-cols-3 gap-3 text-sm">
                      {tab.benefits.map((benefit, idx) => (
                        <BulletListItem
                          key={idx}
                          title={benefit.text}
                          variant="check"
                          color="primary"
                          showPlate={false}
                        />
                      ))}
                    </ul>
                  </div>
                </div>
              </TabsContent>
            ))}
          </div>
        </div>
      </Tabs>
    </div>
  )
}
