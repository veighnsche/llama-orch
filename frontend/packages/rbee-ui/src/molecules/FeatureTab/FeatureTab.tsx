import { TabsTrigger } from '@rbee/ui/atoms/Tabs'
import type * as React from 'react'

export interface FeatureTabProps {
  value: string
  icon: React.ReactNode
  label: string
  mobileLabel?: string
}

/**
 * A feature tab trigger with icon and responsive text labels.
 * Used in tabbed feature sections to switch between different feature categories.
 */
export function FeatureTab({ value, icon, label, mobileLabel }: FeatureTabProps) {
  return (
    <TabsTrigger
      value={value}
      className="flex flex-col sm:flex-row items-center justify-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg text-sm font-medium transition-colors"
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
      {mobileLabel && <span className="text-xs text-muted-foreground block leading-none sm:hidden">{mobileLabel}</span>}
    </TabsTrigger>
  )
}
