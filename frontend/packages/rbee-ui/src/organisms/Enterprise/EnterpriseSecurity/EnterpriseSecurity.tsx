import { SecurityCrate } from '@rbee/ui/molecules'
import { Clock, Eye, KeyRound, Lock, Server, Shield } from 'lucide-react'
import Image from 'next/image'

export function EnterpriseSecurity() {
  return (
    <section aria-labelledby="security-h2" className="relative border-b border-border bg-radial-glow px-6 py-24">
      {/* Decorative background illustration */}
      <Image
        src="/decor/security-mesh.webp"
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt="Abstract dark security mesh with linked nodes and amber highlights, suggesting hash-chains, zero-trust, and time-bounded execution"
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
          <p className="mb-2 text-sm font-medium text-primary/70">Defense-in-Depth</p>
          <h2 id="security-h2" className="mb-4 text-4xl font-bold text-foreground">
            Enterprise-Grade Security
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
            Six specialized security crates harden every layer—from auth and inputs to secrets, auditing, JWT lifecycle,
            and time-bounded execution.
          </p>
        </div>

        {/* Security Crates Grid */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-2">
          <SecurityCrate
            icon={Lock}
            title="auth-min: Zero-Trust Authentication"
            subtitle="The Trickster Guardians"
            intro="Constant-time token checks stop CWE-208 leaks. Fingerprints let you log safely. Bind policies block accidental exposure."
            bullets={[
              'Timing-safe comparison (constant-time)',
              'Token fingerprinting (SHA-256)',
              'Bearer token parsing (RFC 6750)',
              'Bind policy enforcement',
            ]}
            docsHref="/docs/security/auth-min"
          />

          <SecurityCrate
            icon={Eye}
            title="audit-logging: Compliance Engine"
            subtitle="Legally Defensible Proof"
            intro="Append-only audit trail with 32 event types. Hash-chain tamper detection. 7-year retention for GDPR."
            bullets={[
              'Immutable audit trail (append-only)',
              '32 event types across 7 categories',
              'Tamper detection (hash chains)',
              '7-year retention (GDPR)',
            ]}
            docsHref="/docs/security/audit-logging"
          />

          <SecurityCrate
            icon={Shield}
            title="input-validation: First Line of Defense"
            subtitle="Trust No Input"
            intro="Prevents injection and exhaustion. Validates identifiers, prompts, paths—before execution."
            bullets={[
              'SQL injection prevention',
              'Command injection prevention',
              'Path traversal prevention',
              'Resource exhaustion prevention',
            ]}
            docsHref="/docs/security/input-validation"
          />

          <SecurityCrate
            icon={Server}
            title="secrets-management: Credential Guardian"
            subtitle="Never in Environment"
            intro="File-scoped secrets with zeroization and systemd credentials. Timing-safe verification."
            bullets={[
              'File-based loading (not env vars)',
              'Memory zeroization on drop',
              'Permission validation (0600)',
              'Timing-safe verification',
            ]}
            docsHref="/docs/security/secrets-management"
          />

          <SecurityCrate
            icon={KeyRound}
            title="jwt-guardian: Token Lifecycle Manager"
            subtitle="Stateless Yet Secure"
            intro="RS256 signature validation with clock-skew tolerance. Revocation lists and short-lived refresh tokens."
            bullets={[
              'RS256/ES256 signature validation',
              'Clock-skew tolerance (±5 min)',
              'Revocation list (Redis-backed)',
              'Short-lived refresh tokens (15 min)',
            ]}
            docsHref="/docs/security/jwt-guardian"
          />

          <SecurityCrate
            icon={Clock}
            title="deadline-propagation: Performance Enforcer"
            subtitle="Every Millisecond Counts"
            intro="Propagates time budgets end-to-end. Aborts doomed work to protect SLOs."
            bullets={[
              'Deadline propagation (client → worker)',
              'Remaining time calculation',
              'Deadline enforcement (abort if insufficient)',
              'Timeout responses (504 Gateway Timeout)',
            ]}
            docsHref="/docs/security/deadline-propagation"
          />
        </div>

        {/* Security Guarantees */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-8 [animation-delay:200ms]">
          <h3 className="mb-6 text-center text-2xl font-semibold text-foreground">Security Guarantees</h3>
          <div className="grid gap-6 md:grid-cols-3">
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-primary" aria-label="Less than 10 percent timing variance">
                &lt; 10%
              </div>
              <div className="text-sm text-foreground/85">Timing variance (constant-time)</div>
            </div>
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-primary" aria-label="100 percent token fingerprinting">
                100%
              </div>
              <div className="text-sm text-foreground/85">Token fingerprinting (no raw tokens)</div>
            </div>
            <div className="text-center">
              <div className="mb-2 text-3xl font-bold text-primary" aria-label="Zero memory leaks">
                Zero
              </div>
              <div className="text-sm text-foreground/85">Memory leaks (zeroization on drop)</div>
            </div>
          </div>
          <p className="mt-6 text-center text-xs text-muted-foreground">
            Figures represent default crate configurations; tune in policy for your environment.
          </p>
        </div>
      </div>
    </section>
  )
}
