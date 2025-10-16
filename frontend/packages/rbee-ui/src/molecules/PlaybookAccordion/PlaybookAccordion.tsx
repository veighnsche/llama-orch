'use client'

import { Table, TableBody, TableCell, TableRow } from '@rbee/ui/atoms/Table'
import type { IconPlateProps } from '@rbee/ui/molecules'
import { IconPlate } from '@rbee/ui/molecules'
import { type ReactNode, useEffect, useRef } from 'react'

type Severity = 'destructive' | 'primary' | 'chart-2' | 'chart-3'

export const severityBg = {
  destructive: 'bg-destructive',
  primary: 'bg-primary',
  'chart-2': 'bg-chart-2',
  'chart-3': 'bg-chart-3',
} as const

interface SeverityDotProps {
  tone: Severity
}

export function SeverityDot({ tone }: SeverityDotProps) {
  const bg = severityBg[tone]
  return (
    <span
      className={`${bg} opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40 shrink-0 mt-2`}
      aria-hidden="true"
    />
  )
}

interface CheckRowProps {
  severity: Severity
  title: string
  meaning: string
  actionLabel: string
  href: string
  guideLabel?: string
  guideHref?: string
}

export function CheckRow({ severity, title, meaning, actionLabel, href, guideLabel, guideHref }: CheckRowProps) {
  return (
    <TableRow className="animate-in fade-in duration-150">
      <TableCell className="w-[280px]">
        <div className="flex items-start gap-2">
          <SeverityDot tone={severity} />
          <span className="font-medium text-foreground">{title}</span>
        </div>
      </TableCell>
      <TableCell className="w-[380px]">
        <p className="text-sm text-muted-foreground">{meaning}</p>
      </TableCell>
      <TableCell className="w-[200px] text-right">
        <div className="flex gap-3 justify-end">
          <a href={href} className="text-xs underline decoration-dotted hover:decoration-solid text-primary">
            {actionLabel}
          </a>
          {guideLabel && guideHref && (
            <a
              href={guideHref}
              className="text-xs underline decoration-dotted hover:decoration-solid text-muted-foreground hover:text-foreground"
            >
              {guideLabel}
            </a>
          )}
        </div>
      </TableCell>
    </TableRow>
  )
}

interface PlaybookItemProps {
  icon: ReactNode
  color: IconPlateProps['tone']
  title: string
  checkCount: number
  severityDots: ReadonlyArray<Severity>
  description: string
  checks: ReadonlyArray<{
    severity: Severity
    title: string
    meaning: string
    actionLabel: string
    href: string
    guideLabel?: string
    guideHref?: string
  }>
  footer?: React.ReactNode
  illustrationSrc?: string
  onToggle?: (isOpen: boolean) => void
}

export function PlaybookItem({
  icon,
  color,
  title,
  checkCount,
  severityDots,
  description,
  checks,
  footer,
  illustrationSrc,
  onToggle,
}: PlaybookItemProps) {
  const detailsRef = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    const details = detailsRef.current
    if (!details) return

    const handleToggle = () => {
      const isOpen = details.open
      onToggle?.(isOpen)

      // Announce to screen readers
      if (isOpen) {
        const announcement = `${title} opened`
        const liveRegion = document.createElement('div')
        liveRegion.setAttribute('role', 'status')
        liveRegion.setAttribute('aria-live', 'polite')
        liveRegion.className = 'sr-only'
        liveRegion.textContent = announcement
        document.body.appendChild(liveRegion)
        setTimeout(() => document.body.removeChild(liveRegion), 1000)
      }
    }

    details.addEventListener('toggle', handleToggle)
    return () => details.removeEventListener('toggle', handleToggle)
  }, [title, onToggle])

  return (
    <details ref={detailsRef} className="group border-b border-border last:border-b-0">
      <summary
        className="flex items-center justify-between gap-3 cursor-pointer px-5 py-4 hover:bg-muted/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            detailsRef.current?.toggleAttribute('open')
          }
        }}
      >
        <span className="flex items-center gap-3">
          <IconPlate icon={icon} tone={color} size="sm" shape="rounded" />
          <span className="font-semibold text-base md:text-lg text-foreground">{title}</span>
          <span className="text-xs text-muted-foreground">•</span>
          <span className="text-xs text-muted-foreground">{checkCount} checks</span>
        </span>
        <span className="flex items-center gap-2 mr-1" aria-hidden="true">
          {illustrationSrc && <img src={illustrationSrc} alt="" className="hidden md:block w-12 h-12 opacity-90" />}
          <span className="hidden sm:flex items-center gap-1.5">
            {severityDots.slice(0, 3).map((severity, i) => {
              const bg = severityBg[severity]
              return (
                <span
                  key={i}
                  className={`${bg} opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40`}
                />
              )
            })}
            <span className="text-muted-foreground text-xs ml-1">severity</span>
          </span>
          <span className="text-muted-foreground text-xs ml-2 group-open:rotate-180 transition-transform">▾</span>
        </span>
      </summary>
      <div className="px-6 pb-5 animate-in fade-in duration-200 group-open:border-t group-open:border-border/80">
        <p className="text-sm text-muted-foreground/90 mb-4 mt-4 max-w-3xl">{description}</p>
        <Table>
          <TableBody>
            {checks.map((check, i) => (
              <CheckRow key={i} {...check} />
            ))}
          </TableBody>
        </Table>
        {footer && <div className="mt-3 pt-3 border-t border-border/70">{footer}</div>}
      </div>
    </details>
  )
}

export function SeverityLegend() {
  return (
    <div
      className="flex items-center gap-4 px-5 py-2 bg-muted/30 border-b border-border/50 text-xs"
      role="note"
      aria-label="Severity legend"
    >
      <span className="text-muted-foreground font-medium">Severity:</span>
      <span className="flex items-center gap-1.5">
        <span className="bg-destructive opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40" />
        <span className="text-foreground">Critical</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="bg-primary opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40" />
        <span className="text-foreground">High</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="bg-chart-2 opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40" />
        <span className="text-foreground">Medium</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="bg-chart-3 opacity-80 size-1.5 rounded-full outline outline-1 outline-border/40" />
        <span className="text-foreground">Low</span>
      </span>
    </div>
  )
}

interface PlaybookHeaderProps {
  title: string
  description: string
  filterCategories: string[]
  selectedCategories?: string[]
  onFilterToggle?: (category: string) => void
  onExpandAll: () => void
  onCollapseAll: () => void
  allExpanded?: boolean
}

export function PlaybookHeader({
  title,
  description,
  filterCategories,
  selectedCategories = [],
  onFilterToggle,
  onExpandAll,
  onCollapseAll,
  allExpanded = false,
}: PlaybookHeaderProps) {
  return (
    <>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 px-5 py-4 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center justify-center rounded-md border border-transparent bg-secondary text-secondary-foreground px-2 py-0.5 text-xs font-medium">
            {title}
          </span>
          <span className="text-sm font-semibold text-foreground">{description}</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {filterCategories.map((category) => {
            const isSelected = selectedCategories.length === 0 || selectedCategories.includes(category)
            return (
              <button
                key={category}
                onClick={() => onFilterToggle?.(category)}
                className={`px-2.5 py-1 rounded-md text-xs transition-colors ${
                  isSelected ? 'bg-primary text-primary-foreground' : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                }`}
                aria-pressed={isSelected}
              >
                {category}
              </button>
            )
          })}
          <button
            onClick={allExpanded ? onCollapseAll : onExpandAll}
            className="ml-2 px-2.5 py-1 rounded-md text-xs border border-border hover:bg-muted transition-colors hidden md:inline-flex"
          >
            {allExpanded ? 'Collapse all' : 'Expand all'}
          </button>
        </div>
      </div>
      <SeverityLegend />
    </>
  )
}
