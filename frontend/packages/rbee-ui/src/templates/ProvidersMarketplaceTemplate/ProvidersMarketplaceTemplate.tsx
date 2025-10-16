import type { LucideIcon } from 'lucide-react'
import * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersMarketplaceFeatureTile = {
  icon: LucideIcon
  title: string
  description: string
}

export type ProvidersMarketplaceFeatureItem = {
  title: string
  description: string
}

export type ProvidersMarketplaceCommissionExample = {
  label: string
  value: string
}

export type ProvidersMarketplaceTemplateProps = {
  featureTiles: ProvidersMarketplaceFeatureTile[]
  marketplaceFeaturesTitle: string
  marketplaceFeatures: ProvidersMarketplaceFeatureItem[]
  commissionStructureTitle: string
  standardCommissionLabel: string
  standardCommissionValue: string
  standardCommissionDescription: string
  youKeepLabel: string
  youKeepValue: string
  youKeepDescription: string
  exampleTitle: string
  exampleItems: ProvidersMarketplaceCommissionExample[]
  exampleTotalLabel: string
  exampleTotalValue: string
  exampleBadgeText: string
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersMarketplaceTemplate - Marketplace features and commission structure
 *
 * @example
 * ```tsx
 * <ProvidersMarketplaceTemplate
 *   featureTiles={[...]}
 *   marketplaceFeatures={[...]}
 *   // ... other props
 * />
 * ```
 */
export function ProvidersMarketplaceTemplate({
  featureTiles,
  marketplaceFeaturesTitle,
  marketplaceFeatures,
  commissionStructureTitle,
  standardCommissionLabel,
  standardCommissionValue,
  standardCommissionDescription,
  youKeepLabel,
  youKeepValue,
  youKeepDescription,
  exampleTitle,
  exampleItems,
  exampleTotalLabel,
  exampleTotalValue,
  exampleBadgeText,
}: ProvidersMarketplaceTemplateProps) {
  return (
    <div>
      {/* Feature Tiles */}
      <div className="mb-16 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {featureTiles.map((tile, idx) => {
          const Icon = tile.icon
          const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
          return (
            <div
              key={idx}
              className={`animate-in fade-in slide-in-from-bottom-2 ${delays[idx % delays.length]} rounded-2xl border/60 bg-card/60 p-6 text-center backdrop-blur motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60`}
            >
              <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                <Icon className="h-7 w-7 text-primary" aria-hidden="true" />
              </div>
              <h3 className="mb-2 text-base font-semibold text-foreground">{tile.title}</h3>
              <p className="text-sm text-muted-foreground">{tile.description}</p>
            </div>
          )
        })}
      </div>

      {/* Features & Commission Split */}
      <div className="rounded-2xl border bg-gradient-to-b from-card to-background p-8 sm:p-10">
        <div className="grid gap-10 lg:grid-cols-[1.15fr_0.85fr]">
          {/* Left: Marketplace Features */}
          <div className="animate-in fade-in slide-in-from-bottom-2 delay-150 motion-reduce:animate-none">
            <h3 className="mb-6 text-2xl font-bold text-foreground">{marketplaceFeaturesTitle}</h3>
            <div className="space-y-2">
              {marketplaceFeatures.map((feature, idx) => (
                <div
                  key={idx}
                  className="flex gap-3 rounded-lg border border-transparent p-3 transition-colors hover:border-border/70"
                >
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">{feature.title}</div>
                    <div className="text-sm text-muted-foreground">{feature.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Commission Structure */}
          <div className="animate-in fade-in slide-in-from-bottom-2 delay-200 motion-reduce:animate-none">
            <h3 className="mb-6 text-2xl font-bold text-foreground">{commissionStructureTitle}</h3>
            <div className="space-y-4">
              {/* Standard Commission Card */}
              <div className="rounded-xl border bg-background/60 p-6 transition-transform hover:translate-y-0.5">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">{standardCommissionLabel}</div>
                  <div className="tabular-nums text-2xl font-extrabold text-primary">{standardCommissionValue}</div>
                </div>
                <div className="text-sm text-muted-foreground">{standardCommissionDescription}</div>
              </div>

              {/* You Keep Card */}
              <div className="rounded-xl border border-emerald-400/30 bg-emerald-400/10 p-6 transition-transform hover:translate-y-0.5">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-xs uppercase tracking-wide text-emerald-400">{youKeepLabel}</div>
                  <div className="tabular-nums text-2xl font-extrabold text-emerald-400">{youKeepValue}</div>
                </div>
                <div className="text-sm text-emerald-400">{youKeepDescription}</div>
              </div>

              {/* Example Table */}
              <div className="space-y-2 rounded-lg border bg-background/60 p-4 text-sm">
                {exampleItems.map((item, idx) => (
                  <div key={idx} className="flex justify-between">
                    <span className="text-muted-foreground">{item.label}</span>
                    <span className="tabular-nums text-foreground">{item.value}</span>
                  </div>
                ))}
                <div className="border-t border-border pt-2">
                  <div className="flex justify-between font-semibold">
                    <span className="text-foreground">{exampleTotalLabel}</span>
                    <span className="tabular-nums text-primary">{exampleTotalValue}</span>
                  </div>
                </div>
                <div className="mt-2 inline-flex rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
                  {exampleBadgeText}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
