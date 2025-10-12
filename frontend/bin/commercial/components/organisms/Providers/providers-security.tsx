import { Shield, Lock, Eye, FileCheck } from 'lucide-react'

export function ProvidersSecurity() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            Your Security Is Our Priority
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            Enterprise-grade security protects your hardware, data, and earnings.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-chart-3/10">
                <Shield className="h-7 w-7 text-chart-3" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Sandboxed Execution</h3>
                <div className="text-sm text-muted-foreground">Complete isolation</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              All jobs run in isolated sandboxes with no access to your files, network, or personal data. Your system
              stays completely secure.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                No file system access
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                No network access
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                No access to personal data
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Automatic cleanup after jobs
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-chart-3/10">
                <Lock className="h-7 w-7 text-chart-3" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Encrypted Communication</h3>
                <div className="text-sm text-muted-foreground">End-to-end encryption</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              All communication between your GPU and the marketplace is encrypted. Your earnings data and job details
              are protected.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                TLS 1.3 encryption
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Secure payment processing
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Protected earnings data
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Private job details
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-chart-3/10">
                <Eye className="h-7 w-7 text-chart-3" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Malware Scanning</h3>
                <div className="text-sm text-muted-foreground">Automatic protection</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Every job is automatically scanned for malware before execution. Suspicious jobs are blocked and reported.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Real‑time malware detection
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Automatic job blocking
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Threat intelligence updates
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Customer vetting process
              </li>
            </ul>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-chart-3/10">
                <FileCheck className="h-7 w-7 text-chart-3" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Hardware Protection</h3>
                <div className="text-sm text-muted-foreground">Warranty-safe operation</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              Temperature monitoring, automatic cooldown periods, and power limits protect your hardware and maintain
              warranty coverage.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Temperature monitoring
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Automatic cooldown periods
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Power consumption limits
              </li>
              <li className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 rounded-full bg-chart-3" />
                Hardware health monitoring
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 rounded-2xl border border-chart-3/50 bg-chart-3/20 p-8 text-center">
          <p className="text-balance text-lg font-medium text-chart-3">
            Plus: €1M insurance coverage included for all providers. Your hardware is protected.
          </p>
        </div>
      </div>
    </section>
  )
}
