import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, BookOpen, MessageCircle } from 'lucide-react'
import { SectionContainer } from '@/components/molecules'
import Image from 'next/image'
import Link from 'next/link'

export function CTASection() {
  return (
    <SectionContainer maxWidth="6xl" bgVariant="background" title={null}>
      <div className="relative overflow-hidden rounded-2xl border border-border bg-card/40 p-8 sm:p-10 md:p-12 shadow-sm">
        {/* Decorative background */}
        <div aria-hidden className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
          <div className="absolute -top-20 right-0 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
        </div>

        <div className="mx-auto grid max-w-6xl grid-cols-1 gap-8 md:grid-cols-12">
          {/* Left: copy + CTAs */}
          <div className="md:col-span-7 space-y-6">
            {/* Eyebrow badge */}
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-background/70 px-3 py-1 text-xs text-muted-foreground animate-fade-in">
              <span className="h-2 w-2 rounded-full bg-primary" />
              100% Open Source • Self-Hosted
            </div>

            {/* Headline */}
            <h2 className="text-4xl/tight md:text-5xl/tight font-semibold tracking-tight animate-fade-in-up">
              Take Control of Your <span className="text-primary">AI Infrastructure</span> Today.
            </h2>

            {/* Subcopy */}
            <p className="text-lg md:text-xl text-muted-foreground max-w-prose animate-fade-in">
              Join hundreds of users, providers, and enterprises who've chosen independence—no vendor lock-in, no data
              exfiltration.
            </p>

            {/* CTAs */}
            <div
              className="flex flex-col sm:flex-row gap-3 sm:gap-4 animate-fade-in-up"
              aria-describedby="cta-footnote"
            >
              {/* Primary CTA */}
              <Button
                asChild
                size="lg"
                className="h-14 px-8 text-lg bg-primary text-primary-foreground hover:bg-primary/90 font-semibold"
              >
                <Link href="/docs/get-started">
                  Get Started Free
                  <ArrowRight className="ml-2 size-5" />
                </Link>
              </Button>
              {/* Documentation CTA */}
              <Button asChild size="lg" variant="outline" className="h-14 px-8 border-border hover:bg-secondary">
                <Link href="/docs">
                  <BookOpen className="mr-2 size-5" />
                  View Documentation
                </Link>
              </Button>
              {/* Discord CTA */}
              <Button asChild size="lg" variant="outline" className="h-14 px-8 border-border hover:bg-secondary">
                <a href="https://discord.gg/orchyra" target="_blank" rel="noreferrer">
                  <MessageCircle className="mr-2 size-5" />
                  Join Discord
                </a>
              </Button>
            </div>

            {/* Reassurance row */}
            <div id="cta-footnote" className="text-sm text-muted-foreground animate-fade-in">
              100% open source. No credit card required.{' '}
              <span className="text-foreground">Install in ~15 minutes.</span>
            </div>

            {/* Quick install snippet */}
            <div className="animate-fade-in">
              <code className="inline-block rounded-md bg-muted px-3 py-2 text-xs font-mono text-foreground border border-border">
                curl -fsSL https://rbee.sh | bash
              </code>
            </div>

            {/* Quick-proof bullets */}
            <ul className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-3 animate-fade-in">
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                OpenAI-compatible API
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                Multi-GPU / multi-node
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                Sandboxed schedulers
              </li>
            </ul>
          </div>

          {/* Right: visual + mini-metrics */}
          <aside className="md:col-span-5">
            <div className="relative mx-auto max-w-md md:max-w-none animate-fade-in">
              <Image
                src="/images/cta-independence-hero.png"
                width={760}
                height={540}
                priority
                className="rounded-xl border border-border shadow-sm object-cover"
                alt="Isometric 3D illustration showing a secure, self-contained AI infrastructure setup. Central focus on a modular hexagonal server cluster with glowing amber status LEDs arranged in a beehive pattern. Each hexagon unit contains visible GPU cards with cooling fins and power indicators. The cluster sits on a clean workspace surface with subtle ambient occlusion shadows. A translucent protective dome or shield surrounds the infrastructure, symbolizing data sovereignty and security. Warm color palette: honey gold (#F59E0B) accents, charcoal gray (#334155) chassis, soft white (#F1F5F9) highlights. Soft directional lighting from top-left creates depth and dimension. Background features subtle circuit board patterns fading to white, suggesting connectivity without complexity. Clean, modern, technical aesthetic that conveys independence, control, and professional-grade infrastructure. Rendered in a approachable SaaS product illustration style with high detail on metallic surfaces and gentle depth of field."
              />
              <div className="mt-4 grid grid-cols-3 gap-3 text-center">
                <div className="rounded-lg border border-border bg-background/60 p-3">
                  <div className="text-2xl font-semibold text-foreground">15m</div>
                  <div className="text-xs text-muted-foreground">to install</div>
                </div>
                <div className="rounded-lg border border-border bg-background/60 p-3">
                  <div className="text-2xl font-semibold text-foreground">100%</div>
                  <div className="text-xs text-muted-foreground">open source</div>
                </div>
                <div className="rounded-lg border border-border bg-background/60 p-3">
                  <div className="text-2xl font-semibold text-foreground">GPU+</div>
                  <div className="text-xs text-muted-foreground">multi-backend</div>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </SectionContainer>
  )
}
