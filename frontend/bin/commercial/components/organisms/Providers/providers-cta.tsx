import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Zap } from "lucide-react"

export function ProvidersCTA() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background via-amber-950/10 to-background px-6 py-24">
      <div className="mx-auto max-w-4xl text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
          <Zap className="h-4 w-4" />
          Start Earning Today
        </div>

        <h2 className="mb-6 text-balance text-4xl font-bold text-foreground lg:text-5xl">
          Your GPUs Are Losing Money Every Second
        </h2>

        <p className="mb-8 text-balance text-xl text-muted-foreground">
          Stop letting your hardware sit idle. Join 500+ providers earning passive income on the rbee marketplace.
        </p>

        <div className="mb-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary">
            Start Earning Now
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
          <Button size="lg" variant="outline" className="border-border text-foreground hover:bg-secondary bg-transparent">
            View Documentation
          </Button>
        </div>

        <div className="grid gap-6 text-sm text-muted-foreground sm:grid-cols-3">
          <div>
            <div className="mb-1 font-medium text-foreground">Setup Time</div>
            <div>Less than 15 minutes</div>
          </div>
          <div>
            <div className="mb-1 font-medium text-foreground">Commission</div>
            <div>Only 15% (you keep 85%)</div>
          </div>
          <div>
            <div className="mb-1 font-medium text-foreground">Minimum Payout</div>
            <div>€25 (weekly payouts)</div>
          </div>
        </div>
      </div>
    </section>
  )
}
