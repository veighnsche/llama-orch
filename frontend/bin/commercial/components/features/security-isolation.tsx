import { Shield, Lock, CheckCircle2 } from "lucide-react"
import { SectionContainer, IconBox } from "@/components/primitives"

export function SecurityIsolation() {
  return (
    <SectionContainer
      title="Security & Isolation"
      bgVariant="background"
      subtitle="Defense-in-depth architecture with five specialized security crates. Enterprise-grade security for your
            homelab."
    >
            <div className="max-w-5xl mx-auto space-y-8">
          {/* Five Security Crates */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={Shield} color="chart-2" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Five Specialized Security Crates</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Each security concern gets its own Rust crate with focused responsibility. No monolithic security
                  module.
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="font-bold text-foreground mb-2">auth-min</div>
                <div className="text-muted-foreground text-sm">Timing-safe token comparison, zero-trust authentication</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="font-bold text-foreground mb-2">audit-logging</div>
                <div className="text-muted-foreground text-sm">Immutable append-only logs with 7-year retention</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="font-bold text-foreground mb-2">input-validation</div>
                <div className="text-muted-foreground text-sm">Injection prevention, schema validation, sanitization</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="font-bold text-foreground mb-2">secrets-management</div>
                <div className="text-muted-foreground text-sm">Secure credential storage, key rotation, encryption</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="font-bold text-foreground mb-2">deadline-propagation</div>
                <div className="text-muted-foreground text-sm">Timeout enforcement, resource cleanup, cascading shutdown</div>
              </div>
            </div>
          </div>

          {/* Process Isolation */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={Lock} color="chart-3" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Process Isolation</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Each worker runs in its own process. Crashes are isolated. No shared state. Clean shutdown guaranteed.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Sandboxed Execution</div>
                  <div className="text-muted-foreground text-sm">Workers run in isolated processes with no shared memory</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Cascading Shutdown</div>
                  <div className="text-muted-foreground text-sm">
                    rbee-keeper → queen-rbee → rbee-hive → workers (30s timeout, then force-kill)
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">VRAM Cleanup</div>
                  <div className="text-muted-foreground text-sm">
                    Workers free VRAM on shutdown, available for games/other apps
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </SectionContainer>
  )
}
