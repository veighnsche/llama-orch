import { Network, AlertTriangle, Database, Activity, CheckCircle2 } from "lucide-react"

export function ErrorHandling() {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Comprehensive Error Handling
          </h2>
          <p className="text-xl text-muted-foreground leading-relaxed">
            19+ error scenarios with clear messages and actionable suggestions. No cryptic failures.
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Network & Connectivity */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-lg bg-destructive/10 flex items-center justify-center">
                  <Network className="h-5 w-5 text-destructive" />
                </div>
                <h3 className="text-lg font-bold text-foreground">Network & Connectivity</h3>
              </div>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-destructive mt-1">•</span>
                  <span>SSH connection timeout (3 retries with backoff)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-destructive mt-1">•</span>
                  <span>SSH authentication failure with fix suggestions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-destructive mt-1">•</span>
                  <span>HTTP connection failures with retry logic</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-destructive mt-1">•</span>
                  <span>Connection loss during inference (partial results saved)</span>
                </li>
              </ul>
            </div>

            {/* Resource Errors */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <AlertTriangle className="h-5 w-5 text-primary" />
                </div>
                <h3 className="text-lg font-bold text-foreground">Resource Errors</h3>
              </div>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span>Insufficient RAM with model size suggestions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span>VRAM exhausted (FAIL FAST, no CPU fallback)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span>Disk space checks before downloads</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-1">•</span>
                  <span>OOM detection during model loading</span>
                </li>
              </ul>
            </div>

            {/* Model & Backend */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-lg bg-chart-2/10 flex items-center justify-center">
                  <Database className="h-5 w-5 text-chart-2" />
                </div>
                <h3 className="text-lg font-bold text-foreground">Model & Backend</h3>
              </div>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-chart-2 mt-1">•</span>
                  <span>Model not found (404) with Hugging Face link</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-2 mt-1">•</span>
                  <span>Private model (403) with auth token suggestion</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-2 mt-1">•</span>
                  <span>Download failures with resume support</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-2 mt-1">•</span>
                  <span>Backend not available with alternatives</span>
                </li>
              </ul>
            </div>

            {/* Process Lifecycle */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
                  <Activity className="h-5 w-5 text-chart-3" />
                </div>
                <h3 className="text-lg font-bold text-foreground">Process Lifecycle</h3>
              </div>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-chart-3 mt-1">•</span>
                  <span>Worker binary not found with install instructions</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-3 mt-1">•</span>
                  <span>Worker crashes during startup (log suggestions)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-3 mt-1">•</span>
                  <span>Graceful shutdown with active requests</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-chart-3 mt-1">•</span>
                  <span>Force-kill after 30s timeout</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-8 bg-chart-3/10 border border-chart-3/20 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="h-6 w-6 text-chart-3 flex-shrink-0" />
              <div>
                <div className="font-bold text-chart-3 mb-2">Exponential Backoff with Jitter</div>
                <div className="text-muted-foreground text-sm leading-relaxed">
                  All retries use exponential backoff with random jitter (0.5-1.5x) to avoid thundering herd problems.
                  SSH: 3 attempts. HTTP: 3 attempts. Downloads: 6 attempts with resume support.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
