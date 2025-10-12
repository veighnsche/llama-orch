import { TrendingUp, Users, Globe, Shield } from "lucide-react"

export function ProvidersMarketplace() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background to-card px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            How the rbee Marketplace Works
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            A fair, transparent marketplace that connects GPU providers with developers who need compute power.
          </p>
        </div>

        <div className="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <TrendingUp className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Dynamic Pricing</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Set your own rates or use automatic pricing based on market demand
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Users className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Growing Demand</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Thousands of developers need GPU power for AI development
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Globe className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Global Reach</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Your GPUs are discoverable by developers worldwide
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Shield className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Fair Commission</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Only 15% commission. You keep 85% of all earnings
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-12">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="mb-6 text-2xl font-bold text-foreground">Marketplace Features</h3>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Automatic Matching</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Our algorithm matches your GPUs with the right jobs based on requirements and your pricing.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Rating System</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Build your reputation with customer ratings. Higher ratings = more jobs and better rates.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Guaranteed Payments</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Customers pay upfront. You're guaranteed to get paid for every job completed.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Dispute Resolution</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Fair dispute resolution process protects both providers and customers.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="mb-6 text-2xl font-bold text-foreground">Commission Structure</h3>
              <div className="space-y-4">
                <div className="rounded-lg border border-border bg-background/50 p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">Standard Commission</div>
                    <div className="text-2xl font-bold text-primary">15%</div>
                  </div>
                  <div className="text-sm leading-relaxed text-muted-foreground">
                    We take 15% to cover marketplace operations, payment processing, and customer support.
                  </div>
                </div>

                <div className="rounded-lg border border-primary/20 bg-primary/10 p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-sm text-primary">You Keep</div>
                    <div className="text-2xl font-bold text-primary">85%</div>
                  </div>
                  <div className="text-sm leading-relaxed text-primary">
                    You keep 85% of all earnings. No hidden fees. No surprise deductions.
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Example: €100 job</span>
                    <span className="text-foreground">€100.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">rbee commission (15%)</span>
                    <span className="text-foreground">-€15.00</span>
                  </div>
                  <div className="border-t border-border pt-2">
                    <div className="flex justify-between font-medium">
                      <span className="text-foreground">Your earnings</span>
                      <span className="text-primary">€85.00</span>
                    </div>
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
