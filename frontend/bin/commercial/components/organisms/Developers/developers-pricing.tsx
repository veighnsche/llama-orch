import { Button } from '@/components/atoms/Button/Button'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

const tiers = [
  {
    name: 'Home/Lab',
    price: '$0',
    period: 'forever',
    description: 'For solo developers and hobbyists',
    features: [
      'Unlimited GPUs',
      'OpenAI-compatible API',
      'Multi-modal support',
      'Community support',
      '100% open source',
      'llama-orch-utils included',
    ],
    cta: 'Download Now',
    highlighted: false,
  },
  {
    name: 'Team',
    price: '€99',
    period: '/month',
    description: 'For small teams (5-10 devs)',
    features: [
      'Everything in Home/Lab',
      'Web UI management',
      'Team collaboration',
      'Priority support',
      'Rhai script templates',
      'Advanced monitoring',
    ],
    cta: 'Start 30-Day Trial',
    highlighted: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For large teams and enterprises',
    features: [
      'Everything in Team',
      'Dedicated instances',
      'Custom SLAs',
      'White-label option',
      'Enterprise support',
      'On-premises deployment',
    ],
    cta: 'Contact Sales',
    highlighted: false,
  },
]

export function DevelopersPricing() {
  return (
    <section className="border-b border-border bg-secondary py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Start Free. Scale When Ready.
          </h2>
        </div>

        <div className="mx-auto mt-16 grid max-w-6xl gap-8 lg:grid-cols-3">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={cn(
                'relative rounded-lg border p-8 bg-card',
                tier.highlighted ? 'border-primary' : 'border-border',
              )}
            >
              {tier.highlighted && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full border border-primary bg-primary px-4 py-1 text-sm font-medium text-primary-foreground">
                  Most Popular
                </div>
              )}

              <div className="mb-6">
                <h3 className="mb-2 text-xl font-semibold text-card-foreground">{tier.name}</h3>
                <div className="mb-2 flex items-baseline gap-1">
                  <span className="text-4xl font-bold text-card-foreground">{tier.price}</span>
                  <span className="text-muted-foreground">{tier.period}</span>
                </div>
                <p className="text-sm text-muted-foreground">{tier.description}</p>
              </div>

              <ul className="mb-8 space-y-3">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3">
                    <Check className="h-5 w-5 flex-shrink-0 text-primary" />
                    <span className="text-sm text-muted-foreground">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button
                className={cn(
                  'w-full',
                  tier.highlighted
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'border-border bg-card text-foreground hover:bg-muted',
                )}
                variant={tier.highlighted ? 'default' : 'outline'}
              >
                {tier.cta}
              </Button>
            </div>
          ))}
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="text-sm text-muted-foreground">
            All tiers include the full rbee orchestrator. No feature gates. No artificial limits.
          </p>
        </div>
      </div>
    </section>
  )
}
