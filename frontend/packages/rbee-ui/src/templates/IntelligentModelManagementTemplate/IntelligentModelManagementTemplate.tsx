'use client'

import { Alert, Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'
import { Check, CheckCircle2, Database } from 'lucide-react'
import type { ReactNode } from 'react'

export interface ModelSource {
  title: string
  example: string
}

export interface ResourceCheck {
  title: string
  description: string
}

export interface IntelligentModelManagementTemplateProps {
  /** Model catalog title */
  catalogTitle: string
  /** Model catalog description */
  catalogDescription: string
  /** Model download timeline content */
  timelineContent: ReactNode
  /** Model sources (3 items) */
  modelSources: ModelSource[]
  /** Preflight checks title */
  preflightTitle: string
  /** Preflight checks description */
  preflightDescription: string
  /** Resource checks (4 items) */
  resourceChecks: ResourceCheck[]
  /** Alert message */
  alertMessage: string
  /** Custom class name */
  className?: string
}

/**
 * IntelligentModelManagementTemplate - Template for model management features
 *
 * @example
 * ```tsx
 * <IntelligentModelManagementTemplate
 *   title="Intelligent Model Management"
 *   subtitle="Automatic model provisioning, caching, and validation."
 *   catalogTitle="Automatic Model Catalog"
 *   resourceChecks={[...]}
 * />
 * ```
 */
export function IntelligentModelManagementTemplate({
  catalogTitle,
  catalogDescription,
  timelineContent,
  modelSources,
  preflightTitle,
  preflightDescription,
  resourceChecks,
  alertMessage,
  className,
}: IntelligentModelManagementTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-5xl space-y-8">
        {/* Automatic Model Catalog - Full width */}
        <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
          <CardContent className="space-y-6 pt-6">
            <div className="flex items-start gap-4">
              <IconPlate icon={Database} tone="chart-3" size="md" shape="rounded" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">{catalogTitle}</h3>
                <p className="text-muted-foreground leading-relaxed">{catalogDescription}</p>
              </div>
            </div>

            {/* Terminal timeline */}
            <div
              className="bg-background rounded-xl p-6 font-mono text-sm leading-relaxed shadow-sm"
              aria-label="Model download and validation log"
              aria-live="polite"
            >
              {timelineContent}
            </div>

            {/* Feature strip */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
              {modelSources.map((source, idx) => (
                <div
                  key={idx}
                  className="bg-secondary/60 border border-border rounded-lg p-4 hover:-translate-y-0.5 transition-transform"
                >
                  <div className="text-sm font-semibold mb-1 text-chart-3">{source.title}</div>
                  <div className="text-xs text-muted-foreground">{source.example}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Resource Preflight Checks - Full width */}
        <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500 delay-100">
          <CardContent className="space-y-6 pt-6">
            <div className="flex items-start gap-4">
              <IconPlate icon={CheckCircle2} tone="chart-2" size="md" shape="rounded" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">{preflightTitle}</h3>
                <p className="text-muted-foreground leading-relaxed">{preflightDescription}</p>
              </div>
            </div>

            {/* Checklist grid */}
            <div className="grid sm:grid-cols-2 gap-4">
              {resourceChecks.map((check, idx) => (
                <div key={idx} className="bg-background rounded-lg p-4 flex items-start gap-3">
                  <CheckCircle2 className="size-5 text-chart-3 mt-0.5 shrink-0" aria-hidden="true" />
                  <div>
                    <div className="font-semibold text-foreground">{check.title}</div>
                    <div className="text-sm text-muted-foreground">{check.description}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Info bar */}
            <Alert variant="primary">
              <Check />
              {alertMessage}
            </Alert>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
