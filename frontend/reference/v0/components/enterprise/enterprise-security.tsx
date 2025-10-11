import { Shield, Lock, Eye, Server, Clock } from "lucide-react"

export function EnterpriseSecurity() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">Enterprise-Grade Security</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            Five specialized security crates provide defense-in-depth protection against the most sophisticated attacks.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Security Crate 1 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Lock className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">auth-min: Zero-Trust Authentication</h3>
                <p className="text-sm text-slate-400">The Trickster Guardians</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-slate-300">
              Timing-safe token comparison prevents CWE-208 attacks. Token fingerprinting for safe logging. Bind policy
              enforcement prevents accidental exposure.
            </p>

            <div className="space-y-2">
              {[
                "Timing-safe comparison (constant-time)",
                "Token fingerprinting (SHA-256)",
                "Bearer token parsing (RFC 6750)",
                "Bind policy enforcement",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="text-green-400">✓</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Crate 2 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Eye className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">audit-logging: Compliance Engine</h3>
                <p className="text-sm text-slate-400">Legally Defensible Proof</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-slate-300">
              Immutable audit trail with 32 event types. Tamper detection via blockchain-style hash chains. 7-year
              retention for GDPR compliance.
            </p>

            <div className="space-y-2">
              {[
                "Immutable audit trail (append-only)",
                "32 event types across 7 categories",
                "Tamper detection (hash chains)",
                "7-year retention (GDPR)",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="text-green-400">✓</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Crate 3 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Shield className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">input-validation: First Line of Defense</h3>
                <p className="text-sm text-slate-400">Trust No Input</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-slate-300">
              Prevents injection attacks and resource exhaustion. Validates identifiers, model references, prompts, and
              paths before processing.
            </p>

            <div className="space-y-2">
              {[
                "SQL injection prevention",
                "Command injection prevention",
                "Path traversal prevention",
                "Resource exhaustion prevention",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="text-green-400">✓</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Crate 4 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Server className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">secrets-management: Credential Guardian</h3>
                <p className="text-sm text-slate-400">Never in Environment</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-slate-300">
              File-based secrets with memory zeroization. Systemd credentials support. Timing-safe verification prevents
              memory dump attacks.
            </p>

            <div className="space-y-2">
              {[
                "File-based loading (not env vars)",
                "Memory zeroization on drop",
                "Permission validation (0600)",
                "Timing-safe verification",
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                  <span className="text-green-400">✓</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Crate 5 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8 lg:col-span-2">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Clock className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">deadline-propagation: Performance Enforcer</h3>
                <p className="text-sm text-slate-400">Every Millisecond Counts</p>
              </div>
            </div>

            <p className="mb-4 leading-relaxed text-slate-300">
              Ensures rbee never wastes cycles on doomed work. Deadline propagation from client to worker. Aborts
              immediately when deadline exceeded.
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                {["Deadline propagation (client → worker)", "Remaining time calculation"].map((item, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                    <span className="text-green-400">✓</span>
                    <span>{item}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-2">
                {["Deadline enforcement (abort if insufficient)", "Timeout responses (504 Gateway Timeout)"].map(
                  (item, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-slate-400">
                      <span className="text-green-400">✓</span>
                      <span>{item}</span>
                    </div>
                  ),
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Security Guarantees */}
        <div className="mt-12 rounded-lg border border-amber-500/20 bg-amber-500/5 p-8">
          <h3 className="mb-6 text-center text-2xl font-semibold text-white">Security Guarantees</h3>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-amber-400">&lt; 10%</div>
              <div className="text-sm text-slate-400">Timing variance (constant-time)</div>
            </div>
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-amber-400">100%</div>
              <div className="text-sm text-slate-400">Token fingerprinting (no raw tokens)</div>
            </div>
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-amber-400">Zero</div>
              <div className="text-sm text-slate-400">Memory leaks (zeroization on drop)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
