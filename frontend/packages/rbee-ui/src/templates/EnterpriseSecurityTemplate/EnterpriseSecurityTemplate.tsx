import { SecurityCrate } from '@rbee/ui/molecules'
import type { ReactNode } from 'react'
import Image from 'next/image'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type SecurityCrateData = {
  icon: ReactNode
  title: string
  subtitle: string
  intro: string
  bullets: string[]
  docsHref: string
}

export type SecurityGuarantee = {
  value: string
  label: string
  ariaLabel?: string
}

export type EnterpriseSecurityTemplateProps = {
  backgroundImage: {
    src: string
    alt: string
  }
  securityCrates: SecurityCrateData[]
  guarantees: {
    heading: string
    stats: SecurityGuarantee[]
    footnote: string
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseSecurityTemplate({
  backgroundImage,
  securityCrates,
  guarantees,
}: EnterpriseSecurityTemplateProps) {
  return (
    <div className="relative">
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10">
        {/* Security Crates Grid */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-2">
          {securityCrates.map((crate, idx) => (
            <SecurityCrate
              key={idx}
              icon={crate.icon}
              title={crate.title}
              subtitle={crate.subtitle}
              intro={crate.intro}
              bullets={crate.bullets}
              docsHref={crate.docsHref}
            />
          ))}
        </div>

        {/* Security Guarantees */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-8 [animation-delay:200ms]">
          <h3 className="mb-6 text-center text-2xl font-semibold text-foreground">{guarantees.heading}</h3>
          <div className="grid gap-6 md:grid-cols-3">
            {guarantees.stats.map((stat, idx) => (
              <div key={idx} className="text-center">
                <div className="mb-2 text-3xl font-bold text-primary" aria-label={stat.ariaLabel}>
                  {stat.value}
                </div>
                <div className="text-sm text-foreground/85">{stat.label}</div>
              </div>
            ))}
          </div>
          <p className="mt-6 text-center text-xs text-muted-foreground">{guarantees.footnote}</p>
        </div>
      </div>
    </div>
  )
}
