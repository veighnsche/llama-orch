import { DollarSign, Shield, Sliders, Zap } from "lucide-react"

export function ProvidersSolution() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            Turn Your GPUs Into a Revenue Stream
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            rbee connects your idle GPUs with developers who need compute power. You set the price, control
            availability, and earn passive income.
          </p>
        </div>

        <div className="mb-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <DollarSign className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Passive Income</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Earn €50-200/month per GPU while you sleep, game, or work
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Sliders className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Full Control</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Set your own prices, availability windows, and usage limits
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Shield className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Secure & Private</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Your data stays private. Sandboxed execution. No access to your files
            </p>
          </div>

          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Zap className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-foreground">Easy Setup</h3>
            <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
              Install in 10 minutes. No technical expertise required
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-12">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="mb-6 text-2xl font-bold text-foreground">How It Works</h3>
              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                    1
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Install rbee</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      One command installs rbee on your machine. Works on Windows, Mac, and Linux.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                    2
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Configure Your GPUs</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Set your pricing, availability windows, and usage limits through the web dashboard.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                    3
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Join the Marketplace</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Your GPUs appear in the rbee marketplace. Developers can rent your compute power.
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                    4
                  </div>
                  <div>
                    <div className="mb-1 font-medium text-foreground">Earn Passive Income</div>
                    <div className="text-sm leading-relaxed text-muted-foreground">
                      Get paid automatically. Track earnings in real-time. Withdraw anytime.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-center">
              <div className="w-full max-w-md rounded-xl border border-border bg-background p-6">
                <div className="mb-4 text-sm font-medium text-muted-foreground">Example Earnings</div>
                <div className="mb-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-foreground">RTX 4090</div>
                      <div className="text-xs text-muted-foreground">24GB VRAM • 450W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-primary">€180/mo</div>
                      <div className="text-xs text-muted-foreground">at 80% utilization</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-foreground">RTX 4080</div>
                      <div className="text-xs text-muted-foreground">16GB VRAM • 320W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-primary">€140/mo</div>
                      <div className="text-xs text-muted-foreground">at 80% utilization</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-foreground">RTX 3080</div>
                      <div className="text-xs text-muted-foreground">10GB VRAM • 320W</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-primary">€90/mo</div>
                      <div className="text-xs text-muted-foreground">at 80% utilization</div>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border border-primary/20 bg-primary/10 p-4">
                  <div className="text-xs text-primary">
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
