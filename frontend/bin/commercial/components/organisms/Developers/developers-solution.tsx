import { Cpu, DollarSign, Lock, Zap } from "lucide-react"

export function DevelopersSolution() {
  return (
    <section className="border-b border-border py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Your Hardware. Your Models. Your Control.
          </h2>
          <p className="text-balance text-lg leading-relaxed text-muted-foreground">
            rbee orchestrates AI inference across every GPU in your home network—workstations, gaming PCs, Macs—turning
            idle hardware into a private AI infrastructure.
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
          <div className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary/50 hover:bg-card/80">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <DollarSign className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-card-foreground">Zero Ongoing Costs</h3>
            <p className="text-balance text-sm leading-relaxed text-muted-foreground">
              Pay only for electricity. No subscriptions. No per-token fees.
            </p>
          </div>

          <div className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary/50 hover:bg-card/80">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Lock className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-card-foreground">Complete Privacy</h3>
            <p className="text-balance text-sm leading-relaxed text-muted-foreground">
              Code never leaves your network. GDPR-compliant by default.
            </p>
          </div>

          <div className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary/50 hover:bg-card/80">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Zap className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-card-foreground">Never Changes</h3>
            <p className="text-balance text-sm leading-relaxed text-muted-foreground">
              Models update only when YOU decide. No surprise breakages.
            </p>
          </div>

          <div className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary/50 hover:bg-card/80">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Cpu className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-card-foreground">Use All Your Hardware</h3>
            <p className="text-balance text-sm leading-relaxed text-muted-foreground">
              Orchestrate across CUDA, Metal, CPU. Every GPU contributes.
            </p>
          </div>
        </div>

        {/* Architecture Diagram */}
        <div className="mx-auto mt-16 max-w-3xl">
          <div className="rounded-lg border border-border bg-card p-8">
            <h3 className="mb-6 text-center text-xl font-semibold text-card-foreground">The Bee Architecture</h3>
            <div className="flex flex-col items-center gap-6">
              <div className="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-6 py-3">
                <span className="text-2xl">👑</span>
                <div>
                  <div className="font-semibold text-foreground">queen-rbee</div>
                  <div className="text-sm text-muted-foreground">Orchestrator (brain)</div>
                </div>
              </div>

              <div className="h-8 w-px bg-border" />

              <div className="flex items-center gap-3 rounded-lg border border-border bg-muted px-6 py-3">
                <span className="text-2xl">🍯</span>
                <div>
                  <div className="font-semibold text-foreground">rbee-hive</div>
                  <div className="text-sm text-muted-foreground">Resource manager</div>
                </div>
              </div>

              <div className="h-8 w-px bg-border" />

              <div className="grid gap-4 sm:grid-cols-3">
                <div className="flex items-center gap-2 rounded-lg border border-border bg-muted px-4 py-2">
                  <span className="text-xl">🐝</span>
                  <div className="text-sm">
                    <div className="font-semibold text-foreground">Worker 1</div>
                    <div className="text-muted-foreground">CUDA GPU</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 rounded-lg border border-border bg-muted px-4 py-2">
                  <span className="text-xl">🐝</span>
                  <div className="text-sm">
                    <div className="font-semibold text-foreground">Worker 2</div>
                    <div className="text-muted-foreground">Metal GPU</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 rounded-lg border border-border bg-muted px-4 py-2">
                  <span className="text-xl">🐝</span>
                  <div className="text-sm">
                    <div className="font-semibold text-foreground">Worker 3</div>
                    <div className="text-muted-foreground">CPU</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
