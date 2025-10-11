import { Button } from "@/components/ui/button"
import { ArrowRight, Zap } from "lucide-react"

export function ProvidersCTA() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 via-amber-950/10 to-slate-950 px-6 py-24">
      <div className="mx-auto max-w-4xl text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-amber-500/20 bg-amber-500/10 px-4 py-2 text-sm text-amber-400">
          <Zap className="h-4 w-4" />
          Start Earning Today
        </div>

        <h2 className="mb-6 text-balance text-4xl font-bold text-white lg:text-5xl">
          Your GPUs Are Losing Money Every Second
        </h2>

        <p className="mb-8 text-balance text-xl text-slate-300">
          Stop letting your hardware sit idle. Join 500+ providers earning passive income on the rbee marketplace.
        </p>

        <div className="mb-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Button size="lg" className="bg-amber-500 text-slate-950 hover:bg-amber-400">
            Start Earning Now
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
          <Button size="lg" variant="outline" className="border-slate-700 text-white hover:bg-slate-800 bg-transparent">
            View Documentation
          </Button>
        </div>

        <div className="grid gap-6 text-sm text-slate-400 sm:grid-cols-3">
          <div>
            <div className="mb-1 font-medium text-white">Setup Time</div>
            <div>Less than 15 minutes</div>
          </div>
          <div>
            <div className="mb-1 font-medium text-white">Commission</div>
            <div>Only 15% (you keep 85%)</div>
          </div>
          <div>
            <div className="mb-1 font-medium text-white">Minimum Payout</div>
            <div>€25 (weekly payouts)</div>
          </div>
        </div>
      </div>
    </section>
  )
}
