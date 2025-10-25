'use client'

import { PlaybookHeader, PlaybookItem, StatusKPI, TerminalWindow } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'
import { useCallback } from 'react'

export interface PlaybookCheck {
  severity: 'destructive' | 'primary' | 'chart-2' | 'chart-3'
  title: string
  meaning: string
  actionLabel: string
  href: string
  guideLabel: string
  guideHref: string
}

export interface PlaybookCategory {
  icon: ReactNode
  color: 'warning' | 'primary' | 'chart-2' | 'chart-3'
  title: string
  checkCount: number
  severityDots: Array<'destructive' | 'primary' | 'chart-2' | 'chart-3'>
  description: string
  checks: PlaybookCheck[]
  footer?: ReactNode
}

export interface StatusKPIData {
  icon: ReactNode
  color: 'chart-3' | 'primary' | 'chart-2'
  label: string
  value: string
}

export interface ErrorHandlingTemplateProps {
  statusKPIs: StatusKPIData[]
  terminalContent: ReactNode
  terminalFooter?: ReactNode
  playbookCategories: PlaybookCategory[]
  className?: string
}

export function ErrorHandlingTemplate({
  statusKPIs,
  terminalContent,
  terminalFooter,
  playbookCategories,
  className,
}: ErrorHandlingTemplateProps) {
  const handleExpandAll = useCallback(() => {
    document.querySelectorAll<HTMLDetailsElement>('#playbook details').forEach((d) => (d.open = true))
  }, [])

  const handleCollapseAll = useCallback(() => {
    document.querySelectorAll<HTMLDetailsElement>('#playbook details').forEach((d) => (d.open = false))
  }, [])

  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2">
          {statusKPIs.map((kpi, idx) => (
            <StatusKPI key={idx} icon={kpi.icon} color={kpi.color} label={kpi.label} value={kpi.value} />
          ))}
        </div>

        <div className="animate-in fade-in slide-in-from-bottom-2 delay-100">
          <TerminalWindow
            title="error timeline — retries & jitter"
            ariaLabel="Error timeline with retry examples"
            footer={terminalFooter}
          >
            {terminalContent}
          </TerminalWindow>
        </div>

        <div id="playbook" className="rounded border border-border bg-card overflow-hidden">
          <PlaybookHeader
            title="Playbook"
            description="19+ scenarios · 4 categories"
            filterCategories={['Network', 'Resource', 'Model', 'Process']}
            selectedCategories={[]}
            onFilterToggle={() => {}}
            onExpandAll={handleExpandAll}
            onCollapseAll={handleCollapseAll}
          />

          {playbookCategories.map((category, idx) => (
            <PlaybookItem
              key={idx}
              icon={category.icon}
              color={category.color}
              title={category.title}
              checkCount={category.checkCount}
              severityDots={category.severityDots}
              description={category.description}
              checks={category.checks}
              footer={category.footer}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
