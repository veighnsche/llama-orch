import { type ReactNode } from 'react'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface CompliancePillarProps {
  /** Icon element (e.g., Globe, Shield, Lock) */
  icon: ReactNode
  /** Standard name (e.g., "GDPR", "SOC2", "ISO 27001") */
  title: string
  /** Standard type (e.g., "EU Regulation", "US Standard") */
  subtitle: string
  /** List of compliance requirements */
  checklist: string[]
  /** Optional callout content (endpoints, criteria, controls) */
  callout?: ReactNode
  /** Additional CSS classes */
  className?: string
}

/**
 * CompliancePillar molecule for displaying compliance standards
 * with requirements checklist and optional callout
 */
export function CompliancePillar({
  icon,
  title,
  subtitle,
  checklist,
  callout,
  className,
}: CompliancePillarProps) {
  const titleId = `compliance-${title.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <div
      className={cn(
        'h-full rounded-2xl border border-border bg-card/60 p-8',
        'transition-shadow hover:shadow-lg',
        className,
      )}
      aria-labelledby={titleId}
    >
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 p-3 text-primary">
          {icon}
        </div>
        <div>
          <h3 id={titleId} className="text-2xl font-bold text-foreground">
            {title}
          </h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>
      </div>

      {/* Checklist */}
      <ul className="space-y-3" role="list">
        {checklist.map((item, idx) => (
          <li
            key={idx}
            className="flex items-start gap-2"
            role="listitem"
            aria-label={`${title} requirement: ${item}`}
          >
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-chart-3" aria-hidden="true" />
            <span className="text-sm leading-relaxed text-foreground/85">{item}</span>
          </li>
        ))}
      </ul>

      {/* Callout slot */}
      {callout && <div className="mt-6">{callout}</div>}
    </div>
  )
}
