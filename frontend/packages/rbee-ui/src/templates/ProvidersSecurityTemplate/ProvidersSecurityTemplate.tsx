import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { type LucideIcon, Shield } from 'lucide-react'
import * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersSecurityItem = {
  icon: LucideIcon
  title: string
  subtitle?: string
  body: string
  points: string[]
}

export type ProvidersSecurityTemplateProps = {
  items: ProvidersSecurityItem[]
  ribbon?: { text: string }
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersSecurityTemplate - Security features section for GPU providers
 *
 * @example
 * ```tsx
 * <ProvidersSecurityTemplate
 *   items={[...]}
 *   ribbon={{ text: "€1M insurance coverage included" }}
 * />
 * ```
 */
export function ProvidersSecurityTemplate({ items, ribbon }: ProvidersSecurityTemplateProps) {
  return (
    <div>
      {/* Security Cards Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {items.map((item, idx) => {
          const Icon = item.icon
          const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
          return (
            <div
              key={idx}
              className={cn(
                'animate-in fade-in slide-in-from-bottom-2 rounded-2xl border/70 bg-gradient-to-b from-card/70 to-background/60 p-6 backdrop-blur transition-transform hover:translate-y-0.5 motion-reduce:animate-none supports-[backpack-filter]:bg-background/60 sm:p-7',
                delays[idx % delays.length],
              )}
            >
              <div className="mb-5 flex items-center gap-4">
                <IconPlate icon={Icon} size="lg" className="bg-emerald-400/10 text-emerald-400" />
                <div>
                  <h3 className="text-lg font-semibold text-foreground">{item.title}</h3>
                  {item.subtitle && <div className="text-xs text-muted-foreground">{item.subtitle}</div>}
                </div>
              </div>
              <p className="mb-4 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
              <ul className="space-y-2">
                {item.points.map((point, pidx) => (
                  <li key={pidx} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          )
        })}
      </div>

      {/* Insurance Ribbon */}
      {ribbon && (
        <div className="mt-10 rounded-2xl border border-emerald-400/30 bg-emerald-400/10 p-5 text-center">
          <p className="flex items-center justify-center gap-2 text-balance text-base font-medium text-emerald-400 lg:text-lg">
            <Shield className="h-4 w-4" aria-hidden="true" />
            <span className="tabular-nums">{ribbon.text}</span>
          </p>
        </div>
      )}
    </div>
  )
}
