import { AlertTriangle, DollarSign, Lock } from "lucide-react"

export function DevelopersProblem() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-red-950/20 via-slate-950 to-slate-950 py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            The Hidden Risk of AI-Assisted Development
          </h2>
          <p className="text-balance text-lg leading-relaxed text-slate-300">
            You&apos;re building complex codebases with AI assistance. But what happens when your provider changes the
            rules?
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-3">
          <div className="group relative overflow-hidden rounded-lg border border-red-900/50 bg-gradient-to-b from-red-950/50 to-slate-900 p-8 transition-all hover:border-red-800">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
              <AlertTriangle className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-white">The Model Changes</h3>
            <p className="text-balance leading-relaxed text-slate-400">
              Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your
              team is blocked.
            </p>
          </div>

          <div className="group relative overflow-hidden rounded-lg border border-orange-900/50 bg-gradient-to-b from-orange-950/50 to-slate-900 p-8 transition-all hover:border-orange-800">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
              <DollarSign className="h-6 w-6 text-orange-400" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-white">The Price Increases</h3>
            <p className="text-balance leading-relaxed text-slate-400">
              $20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of
              control.
            </p>
          </div>

          <div className="group relative overflow-hidden rounded-lg border border-red-900/50 bg-gradient-to-b from-red-950/50 to-slate-900 p-8 transition-all hover:border-red-800">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
              <Lock className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-white">The Provider Shuts Down</h3>
            <p className="text-balance leading-relaxed text-slate-400">
              API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes
              unmaintainable overnight.
            </p>
          </div>
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="text-balance text-lg font-medium leading-relaxed text-red-400">
            Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external
            providers.
          </p>
        </div>
      </div>
    </section>
  )
}
