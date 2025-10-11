import { Download, Settings, Globe, Wallet } from "lucide-react"

export function ProvidersHowItWorks() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 to-slate-900 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">
            Start Earning in 4 Simple Steps
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            No technical expertise required. Get your GPUs earning in less than 15 minutes.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Download className="h-8 w-8 text-white" />
            </div>
            <div className="mb-2 text-sm font-medium text-amber-400">Step 1</div>
            <h3 className="mb-3 text-xl font-bold text-white">Install rbee</h3>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Download and install rbee with one command. Works on Windows, Mac, and Linux.
            </p>
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
              <code className="text-xs text-amber-400">curl -sSL rbee.dev/install.sh | sh</code>
            </div>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Settings className="h-8 w-8 text-white" />
            </div>
            <div className="mb-2 text-sm font-medium text-amber-400">Step 2</div>
            <h3 className="mb-3 text-xl font-bold text-white">Configure Settings</h3>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Set your pricing, availability windows, and usage limits through the intuitive web dashboard.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                Set hourly rate
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                Define availability
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                Set usage limits
              </li>
            </ul>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Globe className="h-8 w-8 text-white" />
            </div>
            <div className="mb-2 text-sm font-medium text-amber-400">Step 3</div>
            <h3 className="mb-3 text-xl font-bold text-white">Join Marketplace</h3>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute
              power.
            </p>
            <div className="rounded-lg border border-green-900/50 bg-green-950/20 p-3">
              <div className="text-xs font-medium text-green-400">Your GPUs are now live and earning!</div>
            </div>
          </div>

          <div className="relative">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500">
              <Wallet className="h-8 w-8 text-white" />
            </div>
            <div className="mb-2 text-sm font-medium text-amber-400">Step 4</div>
            <h3 className="mb-3 text-xl font-bold text-white">Get Paid</h3>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Track earnings in real-time. Automatic payouts. Withdraw to your bank account or crypto wallet anytime.
            </p>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="flex justify-between">
                <span>Payout frequency:</span>
                <span className="text-white">Weekly</span>
              </div>
              <div className="flex justify-between">
                <span>Minimum payout:</span>
                <span className="text-white">€25</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12 text-center">
          <p className="text-lg text-slate-300">
            Average setup time: <span className="font-bold text-amber-400">12 minutes</span>
          </p>
        </div>
      </div>
    </section>
  )
}
