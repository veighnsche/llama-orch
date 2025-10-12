import { AlertCircle, TrendingDown, Zap } from "lucide-react"

export function ProvidersProblem() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 via-red-950/10 to-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">
            Your Hardware Is Losing Money Every Second
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            That expensive GPU you bought? It sits idle 90% of the time, costing you electricity while earning nothing.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-3">
          <div className="rounded-2xl border border-red-900/50 bg-gradient-to-b from-red-950/20 to-slate-950 p-8">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
              <TrendingDown className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="mb-3 text-xl font-bold text-white">Wasted Investment</h3>
            <p className="text-pretty leading-relaxed text-slate-300">
              You spent €1,500+ on a high-end GPU. It runs at full capacity maybe 10% of the time. The other 90%? Pure
              waste.
            </p>
          </div>

          <div className="rounded-2xl border border-red-900/50 bg-gradient-to-b from-red-950/20 to-slate-950 p-8">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
              <Zap className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="mb-3 text-xl font-bold text-white">Electricity Costs</h3>
            <p className="text-pretty leading-relaxed text-slate-300">
              Your GPU still draws power even when idle. You're paying €10-30/month in electricity for hardware that's
              doing nothing.
            </p>
          </div>

          <div className="rounded-2xl border border-red-900/50 bg-gradient-to-b from-red-950/20 to-slate-950 p-8">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
              <AlertCircle className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="mb-3 text-xl font-bold text-white">Missed Opportunity</h3>
            <p className="text-pretty leading-relaxed text-slate-300">
              Developers need GPU power and are willing to pay for it. Your idle hardware could be earning
              €50-200/month. Instead, it's earning nothing.
            </p>
          </div>
        </div>

        <div className="mt-12 rounded-2xl border border-slate-800 bg-slate-900/50 p-8 text-center">
          <p className="text-balance text-xl font-medium text-slate-200">
            Every hour your GPU sits idle is money left on the table. What if you could turn that waste into passive
            income?
          </p>
        </div>
      </div>
    </section>
  )
}
