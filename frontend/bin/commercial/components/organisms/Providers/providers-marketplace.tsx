import { TrendingUp, Users, Globe, Shield } from 'lucide-react'

export function ProvidersMarketplace() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background via-primary/5 to-card px-6 py-20 lg:py-28">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in slide-in-from-bottom-2 mb-16 text-center motion-reduce:animate-none">
          <div className="mb-2 text-sm font-medium text-primary/80">Why rbee</div>
          <h2 className="mb-4 text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">
            How the rbee Marketplace Works
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
            A fair, transparent marketplace connecting GPU providers with developers.
          </p>
        </div>

        {/* Feature Tiles */}
        <div className="mb-16 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <div className="animate-in fade-in slide-in-from-bottom-2 delay-75 rounded-2xl border border-border/60 bg-card/60 p-6 text-center backdrop-blur motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
              <TrendingUp className="h-7 w-7 text-primary" aria-hidden="true" />
            </div>
            <h3 className="mb-2 text-base font-semibold text-foreground">Dynamic Pricing</h3>
            <p className="text-sm text-muted-foreground">Set your own rate or use auto-pricing.</p>
          </div>

          <div className="animate-in fade-in slide-in-from-bottom-2 delay-150 rounded-2xl border border-border/60 bg-card/60 p-6 text-center backdrop-blur motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
              <Users className="h-7 w-7 text-primary" aria-hidden="true" />
            </div>
            <h3 className="mb-2 text-base font-semibold text-foreground">Growing Demand</h3>
            <p className="text-sm text-muted-foreground">Thousands of AI jobs posted monthly.</p>
          </div>

          <div className="animate-in fade-in slide-in-from-bottom-2 delay-200 rounded-2xl border border-border/60 bg-card/60 p-6 text-center backdrop-blur motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
              <Globe className="h-7 w-7 text-primary" aria-hidden="true" />
            </div>
            <h3 className="mb-2 text-base font-semibold text-foreground">Global Reach</h3>
            <p className="text-sm text-muted-foreground">Your GPUs are discoverable worldwide.</p>
          </div>

          <div className="animate-in fade-in slide-in-from-bottom-2 delay-300 rounded-2xl border border-border/60 bg-card/60 p-6 text-center backdrop-blur motion-reduce:animate-none supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
              <Shield className="h-7 w-7 text-primary" aria-hidden="true" />
            </div>
            <h3 className="mb-2 text-base font-semibold text-foreground">Fair Commission</h3>
            <p className="text-sm text-muted-foreground">Keep 85% of every payout.</p>
          </div>
        </div>

        {/* Features & Commission Split */}
        <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8 sm:p-10">
          <div className="grid gap-10 lg:grid-cols-[1.15fr_0.85fr]">
            {/* Left: Marketplace Features */}
            <div className="animate-in fade-in slide-in-from-bottom-2 delay-150 motion-reduce:animate-none">
              <h3 className="mb-6 text-2xl font-bold text-foreground">Marketplace Features</h3>
              <div className="space-y-2">
                <div className="flex gap-3 rounded-lg border border-transparent p-3 transition-colors hover:border-border/70">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">Automatic Matching</div>
                    <div className="text-sm text-muted-foreground">
                      Jobs match your GPUs based on specs and your pricing.
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 rounded-lg border border-transparent p-3 transition-colors hover:border-border/70">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">Rating System</div>
                    <div className="text-sm text-muted-foreground">
                      Higher ratings unlock more jobs and better rates.
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 rounded-lg border border-transparent p-3 transition-colors hover:border-border/70">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">Guaranteed Payments</div>
                    <div className="text-sm text-muted-foreground">
                      Customers pre-pay. Every completed job is paid.
                    </div>
                  </div>
                </div>

                <div className="flex gap-3 rounded-lg border border-transparent p-3 transition-colors hover:border-border/70">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">Dispute Resolution</div>
                    <div className="text-sm text-muted-foreground">
                      A fair process protects both providers and customers.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Commission Structure */}
            <div className="animate-in fade-in slide-in-from-bottom-2 delay-200 motion-reduce:animate-none">
              <h3 className="mb-6 text-2xl font-bold text-foreground">Commission Structure</h3>
              <div className="space-y-4">
                {/* Standard Commission Card */}
                <div className="rounded-xl border border-border bg-background/60 p-6 transition-transform hover:translate-y-0.5">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Standard Commission</div>
                    <div className="tabular-nums text-2xl font-extrabold text-primary">15%</div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Covers marketplace operations, payouts, and support.
                  </div>
                </div>

                {/* You Keep Card */}
                <div className="rounded-xl border border-emerald-400/30 bg-emerald-400/10 p-6 transition-transform hover:translate-y-0.5">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-xs uppercase tracking-wide text-emerald-400">You Keep</div>
                    <div className="tabular-nums text-2xl font-extrabold text-emerald-400">85%</div>
                  </div>
                  <div className="text-sm text-emerald-400">No hidden fees or surprise deductions.</div>
                </div>

                {/* Example Table */}
                <div className="space-y-2 rounded-lg border border-border bg-background/60 p-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Example job</span>
                    <span className="tabular-nums text-foreground">€100.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">rbee commission (15%)</span>
                    <span className="tabular-nums text-foreground">−€15.00</span>
                  </div>
                  <div className="border-t border-border pt-2">
                    <div className="flex justify-between font-semibold">
                      <span className="text-foreground">Your earnings</span>
                      <span className="tabular-nums text-primary">€85.00</span>
                    </div>
                  </div>
                  <div className="mt-2 inline-flex rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
                    Effective take-home: 85%
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
