import { DollarSign, Shield, Sliders, Zap } from "lucide-react"

export function ProvidersSolution() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">
            Turn Your GPUs Into a Revenue Stream
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            rbee connects your idle GPUs with developers who need compute power. You set the price, control
            availability, and earn passive income.
          </p>
        </div>

        <div className="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <DollarSign className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Passive Income</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Earn €50-200/month per GPU while you sleep, game, or work
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Sliders className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Full Control</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Set your own prices, availability windows, and usage limits
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Shield className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Secure & Private</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Your data stays private. Sandboxed execution. No access to your files
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10">
              <Zap className="h-8 w-8 text-amber-400" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-white">Easy Setup</h3>
            <p className="text-pretty text-sm leading-relaxed text-slate-400">
              Install in 10 minutes. No technical expertise required
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-12">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="mb-6 text-2xl font-bold text-white">How It Works</h3>
              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500 text-sm font-bold text-slate-950">
                    1
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Install rbee</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      One command installs rbee on your machine. Works on Windows, Mac, and Linux.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500 text-sm font-bold text-slate-950">
                    2
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Configure Your GPUs</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Set your pricing, availability windows, and usage limits through the web dashboard.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500 text-sm font-bold text-slate-950">
                    3
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Join the Marketplace</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Your GPUs appear in the rbee marketplace. Developers can rent your compute power.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-500 text-sm font-bold text-slate-950">
                    4
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-white">Earn Passive Income</div>
                    <div className="text-sm leading-relaxed text-slate-400">
                      Get paid automatically. Track earnings in real-time. Withdraw anytime.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-center">
              <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-950 p-6">
                <div className="mb-4 text-sm font-medium text-slate-400">Example Earnings</div>
                <div className="mb-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">RTX 4090</div>
                      <div className="text-xs text-slate-400">24GB VRAM • 450W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-amber-400">€180/mo</div>
                      <div className="text-xs text-slate-400">at 80% utilization</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">RTX 4080</div>
                      <div className="text-xs text-slate-400">16GB VRAM • 320W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-amber-400">€140/mo</div>
                      <div className="text-xs text-slate-400">at 80% utilization</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">RTX 3080</div>
                      <div className="text-xs text-slate-400">10GB VRAM • 320W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-amber-400">€90/mo</div>
                      <div className="text-xs text-slate-400">at 80% utilization</div>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-4">
                  <div className="text-xs text-amber-400">
                    Earnings vary based on demand, your pricing, and availability. These are conservative estimates.
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
