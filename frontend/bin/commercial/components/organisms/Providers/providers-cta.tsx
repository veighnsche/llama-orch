import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Zap, Clock, Shield, Wallet } from 'lucide-react'
import Image from 'next/image'

export function CTASectionProviders() {
  return (
    <section
      aria-labelledby="providers-cta-h2"
      className="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-amber-950/10 to-background px-6 py-24"
    >
      {/* Decorative Background Image - Repositioned to right edge */}
      <Image
        src="/images/gpu-earnings.png"
        width={960}
        height={540}
        className="pointer-events-none absolute -right-32 top-1/2 hidden -translate-y-1/2 opacity-[0.08] lg:block"
        alt="Cinematic macro shot of three modern NVIDIA RTX GPUs stacked vertically with visible cooling fans and RGB accents, emitting warm amber and orange volumetric light rays from their edges; translucent holographic euro currency symbols (€) and AI task tokens with neural network patterns float upward in a gentle arc representing passive income; dark navy blue to black gradient backdrop with subtle hexagonal mesh pattern; shallow depth of field with bokeh effect; dramatic side lighting creating rim light on GPU edges; photorealistic 3D render style; high contrast with deep shadows; premium tech aesthetic; 8K quality; particles of light dust in the air catching the amber glow; emphasis on AI workload monetization not cryptocurrency mining"
        priority={false}
      />

      <div className="relative z-10 mx-auto max-w-4xl text-center">
        {/* Header Block */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 motion-reduce:animate-none">
          <div
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary"
            role="note"
            title="rbee (pronounced are-bee)"
          >
            <Zap className="h-4 w-4" aria-hidden="true" />
            Start earning today
          </div>

          <h2 id="providers-cta-h2" className="mb-6 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            Turn Idle GPUs Into Weekly Payouts
          </h2>

          <p className="mx-auto mb-8 max-w-2xl text-balance text-lg text-muted-foreground">
            Join 500+ providers monetizing spare GPU time on the rbee marketplace.
          </p>
        </div>

        {/* CTA Actions */}
        <div
          className="animate-in fade-in-50 [animation-delay:120ms] mb-4 flex flex-col items-center justify-center gap-3 motion-reduce:animate-none sm:flex-row sm:gap-4"
          aria-describedby="providers-cta-reassurance"
        >
          <Button
            size="lg"
            className="w-full bg-primary text-primary-foreground hover:bg-primary/90 sm:w-auto"
            aria-label="Start earning now — setup under 15 minutes"
          >
            Start Earning Now
            <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="w-full border-border bg-transparent text-foreground hover:bg-secondary sm:w-auto"
            aria-label="View documentation for providers"
          >
            View Docs
          </Button>
        </div>

        {/* Micro-credibility */}
        <p className="mb-10 text-xs text-muted-foreground/70">
          Data from verified providers; earnings vary by GPU, uptime, and demand.
        </p>

        {/* Reassurance Bar */}
        <div
          id="providers-cta-reassurance"
          className="animate-in fade-in-50 [animation-delay:200ms] mt-10 grid gap-5 text-sm text-muted-foreground motion-reduce:animate-none sm:grid-cols-3"
        >
          <div className="rounded-xl border border-border/60 bg-card/40 p-4">
            <div className="mb-2 flex justify-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <Clock className="h-4 w-4 text-primary" aria-hidden="true" />
              </div>
            </div>
            <div className="font-medium text-foreground">&lt; 15 minutes</div>
            <div className="text-xs">Setup time</div>
          </div>

          <div className="rounded-xl border border-border/60 bg-card/40 p-4">
            <div className="mb-2 flex justify-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-4 w-4 text-primary" aria-hidden="true" />
              </div>
            </div>
            <div className="font-medium text-foreground">15% platform fee</div>
            <div className="text-xs">You keep 85%</div>
          </div>

          <div className="rounded-xl border border-border/60 bg-card/40 p-4">
            <div className="mb-2 flex justify-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <Wallet className="h-4 w-4 text-primary" aria-hidden="true" />
              </div>
            </div>
            <div className="font-medium text-foreground">€25 minimum</div>
            <div className="text-xs">Weekly payouts</div>
          </div>
        </div>
      </div>
    </section>
  )
}

// Export alias for backward compatibility
export const ProvidersCTA = CTASectionProviders
