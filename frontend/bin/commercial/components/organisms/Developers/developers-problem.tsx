import { AlertTriangle, DollarSign, Lock } from 'lucide-react'

export function DevelopersProblem() {
  return (
    <section className="border-b border-border bg-background py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            The Hidden Risk of AI-Assisted Development
          </h2>
          <p className="text-balance text-lg leading-relaxed text-muted-foreground">
            You&apos;re building complex codebases with AI assistance. But what happens when your provider changes the
            rules?
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-3">
          <div className="group relative overflow-hidden rounded-lg border border-destructive/50 bg-card p-8 transition-all hover:border-destructive">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <AlertTriangle className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-card-foreground">The Model Changes</h3>
            <p className="text-balance leading-relaxed text-muted-foreground">
              Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your
              team is blocked.
            </p>
          </div>

          <div className="group relative overflow-hidden rounded-lg border border-primary/50 bg-card p-8 transition-all hover:border-primary">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <DollarSign className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-card-foreground">The Price Increases</h3>
            <p className="text-balance leading-relaxed text-muted-foreground">
              $20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of
              control.
            </p>
          </div>

          <div className="group relative overflow-hidden rounded-lg border border-destructive/50 bg-card p-8 transition-all hover:border-destructive">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
              <Lock className="h-6 w-6 text-destructive" />
            </div>
            <h3 className="mb-3 text-xl font-semibold text-card-foreground">The Provider Shuts Down</h3>
            <p className="text-balance leading-relaxed text-muted-foreground">
              API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes
              unmaintainable overnight.
            </p>
          </div>
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="text-balance text-lg font-medium leading-relaxed text-destructive">
            Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external
            providers.
          </p>
        </div>
      </div>
    </section>
  )
}
