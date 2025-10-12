import { Button } from '@/components/atoms/Button/Button'
import { Calendar, FileText, MessageSquare } from "lucide-react"

export function EnterpriseCTA() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background via-primary/5 to-background px-6 py-24">
      <div className="mx-auto max-w-4xl text-center">
        <h2 className="mb-4 text-4xl font-bold text-foreground lg:text-5xl">Ready to Meet Your Compliance Requirements?</h2>
        <p className="mb-12 text-balance text-xl text-muted-foreground">
          Schedule a demo with our compliance team or download our compliance documentation package.
        </p>

        <div className="mb-12 grid gap-6 md:grid-cols-3">
          {/* Option 1 */}
          <div className="rounded-lg border border-border bg-card p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Calendar className="h-6 w-6 text-primary" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-foreground">Schedule Demo</h3>
            <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
              30-minute demo with our compliance team. See rbee in action.
            </p>
            <Button className="w-full bg-primary text-primary-foreground hover:bg-primary/90">Book Demo</Button>
          </div>

          {/* Option 2 */}
          <div className="rounded-lg border border-border bg-card p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <FileText className="h-6 w-6 text-primary" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-foreground">Compliance Pack</h3>
            <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
              Download GDPR, SOC2, and ISO 27001 documentation.
            </p>
            <Button variant="outline" className="w-full border-border text-foreground hover:bg-secondary bg-transparent">
              Download Docs
            </Button>
          </div>

          {/* Option 3 */}
          <div className="rounded-lg border border-border bg-card p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <MessageSquare className="h-6 w-6 text-primary" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-foreground">Talk to Sales</h3>
            <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
              Discuss your specific compliance requirements.
            </p>
            <Button variant="outline" className="w-full border-border text-foreground hover:bg-secondary bg-transparent">
              Contact Sales
            </Button>
          </div>
        </div>

        <p className="text-sm text-muted-foreground">
          Enterprise support available 24/7. Typical deployment: 6-8 weeks from consultation to production.
        </p>
      </div>
    </section>
  )
}
