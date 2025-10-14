import { AlertCircle, TrendingDown, Zap } from 'lucide-react'
import { ProblemSection } from '@rbee/ui/organisms/ProblemSection'

/**
 * Backward-compatible wrapper for the GPU providers page.
 * Re-exports the shared ProblemSection with provider-specific defaults.
 */
export function ProvidersProblem() {
  return (
    <ProblemSection
      kicker="The Cost of Idle GPUs"
      title="Stop Letting Your Hardware Bleed Money"
      subtitle="Most GPUs sit idle ~90% of the time. They still draw power—and earn nothing."
      items={[
        {
          icon: <TrendingDown className="h-6 w-6" />,
          title: 'Wasted Investment',
          body: 'You paid €1,500+ for a high-end GPU. It\'s busy maybe 10% of the time—the other 90% earns €0.',
          tag: 'Potential earnings €50-200/mo',
          tone: 'destructive',
        },
        {
          icon: <Zap className="h-6 w-6" />,
          title: 'Electricity Costs',
          body: 'Idle GPUs still pull power. That\'s roughly €10-30 each month spent on doing nothing.',
          tag: 'Direct loss €10-30/mo',
          tone: 'destructive',
        },
        {
          icon: <AlertCircle className="h-6 w-6" />,
          title: 'Missed Opportunity',
          body: 'Developers rent GPU power every day. Your machine could join the marketplace and get paid automatically.',
          tag: 'Unrealized €50-200/mo',
          tone: 'destructive',
        },
      ]}
      ctaPrimary={{ label: 'Start Earning', href: '/signup' }}
      ctaSecondary={{ label: 'Estimate My Payout', href: '#earnings-calculator' }}
      ctaCopy="Every idle hour is money left on the table. Turn that waste into passive income with rbee."
    />
  )
}
