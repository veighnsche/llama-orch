import { TrendingUp, Users, Globe, Shield } from "lucide-react"

export function ProvidersMarketplace() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 to-slate-900 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">
            How the rbee Marketplace Works
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            A fair, transparent marketplace that connects GPU providers with developers who need compute power.
          </p>
        </div>

        <div className="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <TrendingUp className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Dynamic Pricing</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Set your own rates or use automatic pricing based on market demand
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Users className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Growing Demand</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Thousands of developers need GPU power for AI development
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Globe className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Global Reach</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Your GPUs are discoverable by developers worldwide
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Shield className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Fair Commission</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Only 15% commission. You keep 85% of all earnings
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-12">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="mb-6 text-2xl font-bold text-white">Marketplace Features</h3>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500/10">
                    <div className="h-2 w-2 rounded-full bg-amber-400" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Automatic Matching</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Our algorithm matches your GPUs with the right jobs based on requirements and your pricing.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500/10">
                    <div className="h-2 w-2 rounded-full bg-amber-400" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Rating System</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Build your reputation with customer ratings. Higher ratings = more jobs and better rates.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500/10">
                    <div className="h-2 w-2 rounded-full bg-amber-400" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Guaranteed Payments</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Customers pay upfront. You're guaranteed to get paid for every job completed.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500/10">
                    <div className="h-2 w-2 rounded-full bg-amber-400" />
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Dispute Resolution</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Fair dispute resolution process protects both providers and customers.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="mb-6 text-2xl font-bold text-white">Commission Structure</h3>
              <div className="space-y-4">
                <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-sm text-slate-400">Standard Commission</div>
                    <div className="text-2xl font-bold text-amber-400">15%</div>
                  </div>
                  <div className="text-sm leading-relaxed text-slate-400">
                    We take 15% to cover marketplace operations, payment processing, and customer support.
                  </div>
                </div>

                <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-sm text-amber-400">You Keep</div>
                    <div className="text-2xl font-bold text-amber-400">85%</div>
                  </div>
                  <div className="text-sm leading-relaxed text-amber-300">
                    You keep 85% of all earnings. No hidden fees. No surprise deductions.
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Example: €100 job</span>
                    <span className="text-white">€100.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">rbee commission (15%)</span>
                    <span className="text-white">-€15.00</span>
                  </div>
                  <div className="border-t border-slate-800 pt-2">
                    <div className="flex justify-between font-medium">
                      <span className="text-white">Your earnings</span>
                      <span className="text-amber-400">€85.00</span>
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
