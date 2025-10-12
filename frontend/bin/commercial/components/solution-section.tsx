import { Anchor, DollarSign, Laptop, Shield } from "lucide-react"

export function SolutionSection() {
  return (
    <section className="py-24 bg-secondary">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Your Hardware. Your Models. <span className="text-primary">Your Control.</span>
          </h2>
          <p className="text-xl text-muted-foreground leading-relaxed text-pretty">
            rbee orchestrates AI inference across every GPU in your home network—workstations, gaming PCs, Macs—turning
            idle hardware into a private AI infrastructure.
          </p>
        </div>

        {/* Architecture Diagram */}
        <div className="max-w-4xl mx-auto mb-16">
          <div className="bg-card border border-border rounded-lg p-8 shadow-lg">
            <div className="text-center mb-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-primary text-sm font-medium">
                The Bee Architecture
              </div>
            </div>

            <div className="space-y-8">
              {/* Queen */}
              <div className="flex flex-col items-center">
                <div className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-bold text-lg shadow-md">
                  👑 Queen-rbee (Orchestrator)
                </div>
                <div className="h-8 w-0.5 bg-border my-2"></div>
              </div>

              {/* Hive Managers */}
              <div className="flex justify-center gap-4">
                <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
                  🍯 Hive Manager 1
                </div>
                <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
                  🍯 Hive Manager 2
                </div>
                <div className="bg-primary/10 text-primary px-4 py-2 rounded-lg font-medium text-sm border border-primary/20">
                  🍯 Hive Manager 3
                </div>
              </div>

              <div className="flex justify-center gap-4">
                <div className="h-8 w-0.5 bg-border"></div>
                <div className="h-8 w-0.5 bg-border"></div>
                <div className="h-8 w-0.5 bg-border"></div>
              </div>

              {/* Workers */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                  🐝 Worker (CUDA)
                </div>
                <div className="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                  🐝 Worker (Metal)
                </div>
                <div className="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                  🐝 Worker (CPU)
                </div>
                <div className="bg-muted text-muted-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center">
                  🐝 Worker (CUDA)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Key Benefits */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          <div className="bg-card border border-border rounded-lg p-6 space-y-3">
            <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
              <DollarSign className="h-5 w-5 text-chart-3 stroke-[3]" />
            </div>
            <h3 className="text-lg font-bold text-card-foreground">Zero Ongoing Costs</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Pay only for electricity. No subscriptions. No per-token fees.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-3">
            <div className="h-10 w-10 rounded-lg bg-chart-2/10 flex items-center justify-center">
              <Shield className="h-5 w-5 text-chart-2" />
            </div>
            <h3 className="text-lg font-bold text-card-foreground">Complete Privacy</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Code never leaves your network. GDPR-compliant by default.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-3">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Anchor className="h-5 w-5 text-primary" />
            </div>
            <h3 className="text-lg font-bold text-card-foreground">Never Changes</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Models update only when YOU decide. No surprise breakages.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-3">
            <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center">
              <Laptop className="h-5 w-5 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-bold text-card-foreground">Use All Your Hardware</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Orchestrate across CUDA, Metal, CPU. Every GPU contributes.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
