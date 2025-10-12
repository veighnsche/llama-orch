import { Download, Settings, Globe, Wallet } from "lucide-react"

export function ProvidersHowItWorks() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background to-card px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            Start Earning in 4 Simple Steps
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            No technical expertise required. Get your GPUs earning in less than 15 minutes.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Download className="h-8 w-8 text-foreground" />
            </div>
            <div className="mb-2 text-sm font-medium text-primary">Step 1</div>
            <h3 className="mb-3 text-xl font-bold text-foreground">Install rbee</h3>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Download and install rbee with one command. Works on Windows, Mac, and Linux.
            </p>
            <div className="rounded-lg border border-border bg-background p-3">
              <code className="text-xs text-primary">curl -sSL rbee.dev/install.sh | sh</code>
            </div>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Settings className="h-8 w-8 text-foreground" />
            </div>
            <div className="mb-2 text-sm font-medium text-primary">Step 2</div>
            <h3 className="mb-3 text-xl font-bold text-foreground">Configure Settings</h3>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Set your pricing, availability windows, and usage limits through the intuitive web dashboard.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Set hourly rate
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Define availability
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                Set usage limits
              </li>
            </ul>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Globe className="h-8 w-8 text-foreground" />
            </div>
            <div className="mb-2 text-sm font-medium text-primary">Step 3</div>
            <h3 className="mb-3 text-xl font-bold text-foreground">Join Marketplace</h3>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute
              power.
            </p>
            <div className="rounded-lg border border-chart-3/50 bg-chart-3/20 p-3">
              <div className="text-xs font-medium text-chart-3">Your GPUs are now live and earning!</div>
            </div>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Wallet className="h-8 w-8 text-foreground" />
            </div>
            <div className="mb-2 text-sm font-medium text-primary">Step 4</div>
            <h3 className="mb-3 text-xl font-bold text-foreground">Get Paid</h3>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Track earnings in real-time. Automatic payouts. Withdraw to your bank account or crypto wallet anytime.
            </p>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex justify-between">
                <span>Payout frequency:</span>
                <span className="text-foreground">Weekly</span>
              </div>
              <div className="flex justify-between">
                <span>Minimum payout:</span>
                <span className="text-foreground">€25</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12 text-center">
          <p className="text-lg text-muted-foreground">
            Average setup time: <span className="font-bold text-primary">12 minutes</span>
          </p>
        </div>
      </div>
    </section>
  )
}
