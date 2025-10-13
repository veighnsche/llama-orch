import { Button } from '@/components/atoms/Button/Button'
import { CTAOptionCard } from '@/components/molecules/CTAOptionCard/CTAOptionCard'
import { Calendar, FileText, MessageSquare } from 'lucide-react'
import Link from 'next/link'
import { TESTIMONIAL_STATS } from '@/data/testimonials'

export function EnterpriseCTA() {
  return (
    <section
      aria-labelledby="cta-h2"
      className="relative border-b border-border bg-gradient-to-b from-background via-primary/5 to-background px-6 py-24 overflow-hidden"
    >
      {/* Decorative Gradient */}
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(60rem_40rem_at_50%_-10%,theme(colors.primary/6),transparent)]"
        aria-hidden="true"
      />

      <div className="relative mx-auto max-w-5xl">
        {/* Header Block */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2 duration-500">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-primary">Get Audit-Ready</p>
          <h2 id="cta-h2" className="mb-4 text-4xl font-bold text-foreground lg:text-5xl">
            Ready to Meet Your Compliance Requirements?
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Book a demo with our compliance team, or download the documentation pack.
          </p>
        </div>

        {/* Trust Strip */}
        <div className="mb-12 grid gap-6 sm:grid-cols-4 text-center">
          {TESTIMONIAL_STATS.map((stat) => (
            <div key={stat.id} className="text-sm">
              <div className="font-semibold text-foreground">{stat.value}</div>
              <div className="text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* CTA Options Grid */}
        <div className="mb-12 grid gap-6 md:grid-cols-3 animate-in fade-in-50" style={{ animationDelay: '120ms' }}>
          {/* Primary: Schedule Demo */}
          <CTAOptionCard
            icon={<Calendar className="h-6 w-6" />}
            title="Schedule Demo"
            body="30-minute demo with our compliance team. See rbee in action."
            tone="primary"
            note="30-minute session • live environment"
            action={
              <Button asChild size="lg" className="w-full" aria-label="Book a 30-minute demo">
                <Link href="/enterprise/demo">Book Demo</Link>
              </Button>
            }
          />

          {/* Secondary: Compliance Pack */}
          <CTAOptionCard
            icon={<FileText className="h-6 w-6" />}
            title="Compliance Pack"
            body="Download GDPR, SOC2, and ISO 27001 documentation."
            note="GDPR, SOC2, ISO 27001 summaries"
            action={
              <Button
                asChild
                variant="outline"
                size="lg"
                className="w-full"
                aria-label="Download compliance documentation pack"
              >
                <Link href="/docs/compliance-pack">Download Docs</Link>
              </Button>
            }
          />

          {/* Tertiary: Talk to Sales */}
          <CTAOptionCard
            icon={<MessageSquare className="h-6 w-6" />}
            title="Talk to Sales"
            body="Discuss your specific compliance requirements."
            note="Share requirements & timelines"
            action={
              <Button
                asChild
                variant="outline"
                size="lg"
                className="w-full"
                aria-label="Contact sales team"
              >
                <Link href="/contact/sales">Contact Sales</Link>
              </Button>
            }
          />
        </div>

        {/* Footer Caption */}
        <p className="text-center text-sm text-muted-foreground">
          Enterprise support 24/7 • Typical deployment: 6–8 weeks from consultation to production.
        </p>
      </div>
    </section>
  )
}
