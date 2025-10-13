import Image from 'next/image'
import { Button, Badge } from '@/components/atoms'
import { Check } from 'lucide-react'

export function PricingHero() {
  return (
    <section
      aria-labelledby="pricing-hero-title"
      className="relative overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-background"
    >
      {/* Visual illustration - full width background */}
      <div className="relative w-full pt-12 lg:pt-16">
        <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-slate-950 via-slate-950/50 to-transparent z-10 pointer-events-none" />
        <Image
          src="/illustrations/pricing-scale-visual.svg"
          alt=""
          width={1400}
          height={500}
          priority
          className="w-full h-auto"
        />
      </div>

      {/* Content overlay */}
      <div className="relative -mt-32 lg:-mt-48 pb-20 lg:pb-28">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            {/* Eyebrow badge */}
            <Badge variant="secondary" className="mb-6 animate-in slide-in-from-bottom-2 delay-100">
              Honest Pricing
            </Badge>

            {/* Headline */}
            <h1
              id="pricing-hero-title"
              className="text-5xl lg:text-7xl font-bold tracking-tight text-white mb-6 leading-tight animate-in slide-in-from-bottom-2 delay-100"
            >
              Start Free.
              <br />
              <span className="text-primary">Scale When Ready.</span>
            </h1>

            {/* Support copy */}
            <p className="text-lg lg:text-xl text-slate-300 mb-10 leading-relaxed max-w-2xl mx-auto animate-in slide-in-from-bottom-2 delay-200">
              Every tier ships the full rbee orchestrator—no feature gates, no artificial limits. OpenAI-compatible
              API, same power on day one. Pay only when you grow.
            </p>

            {/* CTAs */}
            <div className="flex items-center justify-center gap-4 mb-12 animate-in slide-in-from-bottom-2 delay-200">
              <Button size="lg" className="px-8 h-12" aria-label="View Plans">
                View Plans
              </Button>
              <Button 
                variant="ghost" 
                size="lg" 
                className="px-8 h-12 text-white border-white/20 hover:text-white hover:bg-white/10 hover:border-white/30" 
                aria-label="Talk to Sales"
              >
                Talk to Sales
              </Button>
            </div>

            {/* Assurance row */}
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-slate-400 max-w-xl mx-auto animate-in slide-in-from-bottom-2 delay-300">
              {[
                'Full orchestrator on every tier',
                'No feature gates or limits',
                'OpenAI-compatible API',
                'Cancel anytime',
              ].map((item) => (
                <li key={item} className="flex items-center justify-center sm:justify-start gap-2">
                  <Check className="h-4 w-4 text-primary shrink-0" aria-hidden="true" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
