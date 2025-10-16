import { Alert, AlertDescription } from '@rbee/ui/atoms/Alert'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { BulletListItem } from '@rbee/ui/molecules/BulletListItem'
import { Check, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export interface TabConfig {
  value: string
  icon: LucideIcon
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
  sectionId?: string
  bgClassName?: string
}

export function FeaturesTabs({
  title,
  description,
  tabs,
  defaultTab,
  sectionId = 'feature-list',
  bgClassName = 'bg-gradient-to-b from-secondary to-background',
}: FeaturesTabsProps) {
  return (
    <section id={sectionId} className={`py-24 ${bgClassName}`}>
      <div className="container mx-auto px-4">
        <Tabs defaultValue={defaultTab || tabs[0]?.value} className="w-full" orientation="horizontal">
          <div className="grid gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
            {/* Left rail: sticky intro + TabsList */}
            <div className="lg:sticky lg:top-24 self-start space-y-6">
              <div>
                <h2 className="text-3xl font-bold tracking-tight text-foreground">{title}</h2>
                <p className="mt-2 text-muted-foreground">{description}</p>
              </div>

              <TabsList aria-label="Core features">
                {tabs.map((tab) => (
                  <TabsTrigger
                    key={tab.value}
                    value={tab.value}
                    className="flex-col lg:flex-row items-start lg:items-center"
                  >
                    <span className="flex items-center gap-3 w-full">
                      <tab.icon className="size-4 text-muted-foreground group-data-[state=active]:text-primary" />
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
                      <p className="text-muted-foreground leading-relaxed">{tab.description}</p>
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
    </section>
  )
}
