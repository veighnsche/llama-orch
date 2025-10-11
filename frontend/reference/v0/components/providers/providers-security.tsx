import { Shield, Lock, Eye, FileCheck } from "lucide-react"

export function ProvidersSecurity() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">Your Security Is Our Priority</h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            Enterprise-grade security protects your hardware, data, and earnings.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
                <Shield className="h-7 w-7 text-green-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Sandboxed Execution</h3>
                <div className="text-sm text-slate-400">Complete isolation</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              All jobs run in isolated sandboxes with no access to your files, network, or personal data. Your system
              stays completely secure.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                No file system access
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                No network access
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                No access to personal data
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Automatic cleanup after jobs
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
                <Lock className="h-7 w-7 text-green-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Encrypted Communication</h3>
                <div className="text-sm text-slate-400">End-to-end encryption</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              All communication between your GPU and the marketplace is encrypted. Your earnings data and job details
              are protected.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                TLS 1.3 encryption
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Secure payment processing
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Protected earnings data
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Private job details
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
                <Eye className="h-7 w-7 text-green-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Malware Scanning</h3>
                <div className="text-sm text-slate-400">Automatic protection</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Every job is automatically scanned for malware before execution. Suspicious jobs are blocked and reported.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Real-time malware detection
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Automatic job blocking
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Threat intelligence updates
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Customer vetting process
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-green-500/10">
                <FileCheck className="h-7 w-7 text-green-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Hardware Protection</h3>
                <div className="text-sm text-slate-400">Warranty-safe operation</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              Temperature monitoring, automatic cooldown periods, and power limits protect your hardware and maintain
              warranty coverage.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Temperature monitoring
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Automatic cooldown periods
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Power consumption limits
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Hardware health monitoring
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 rounded-2xl border border-green-900/50 bg-green-950/20 p-8 text-center">
          <p className="text-balance text-lg font-medium text-green-300">
            Plus: €1M insurance coverage included for all providers. Your hardware is protected.
          </p>
        </div>
      </div>
    </section>
  )
}
