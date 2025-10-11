import { Shield, Lock, CheckCircle2 } from "lucide-react"

export function SecurityIsolation() {
  return (
    <section className="py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">Security & Isolation</h2>
          <p className="text-xl text-slate-600 leading-relaxed">
            Defense-in-depth architecture with five specialized security crates. Enterprise-grade security for your
            homelab.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-8">
          {/* Five Security Crates */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
                <Shield className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900 mb-2">Five Specialized Security Crates</h3>
                <p className="text-slate-600 leading-relaxed">
                  Each security concern gets its own Rust crate with focused responsibility. No monolithic security
                  module.
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="font-bold text-slate-900 mb-2">auth-min</div>
                <div className="text-slate-600 text-sm">Timing-safe token comparison, zero-trust authentication</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="font-bold text-slate-900 mb-2">audit-logging</div>
                <div className="text-slate-600 text-sm">Immutable append-only logs with 7-year retention</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="font-bold text-slate-900 mb-2">input-validation</div>
                <div className="text-slate-600 text-sm">Injection prevention, schema validation, sanitization</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="font-bold text-slate-900 mb-2">secrets-management</div>
                <div className="text-slate-600 text-sm">Secure credential storage, key rotation, encryption</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="font-bold text-slate-900 mb-2">deadline-propagation</div>
                <div className="text-slate-600 text-sm">Timeout enforcement, resource cleanup, cascading shutdown</div>
              </div>
            </div>
          </div>

          {/* Process Isolation */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center flex-shrink-0">
                <Lock className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900 mb-2">Process Isolation</h3>
                <p className="text-slate-600 leading-relaxed">
                  Each worker runs in its own process. Crashes are isolated. No shared state. Clean shutdown guaranteed.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">Sandboxed Execution</div>
                  <div className="text-slate-600 text-sm">Workers run in isolated processes with no shared memory</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">Cascading Shutdown</div>
                  <div className="text-slate-600 text-sm">
                    rbee-keeper → queen-rbee → rbee-hive → workers (30s timeout, then force-kill)
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">VRAM Cleanup</div>
                  <div className="text-slate-600 text-sm">
                    Workers free VRAM on shutdown, available for games/other apps
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
