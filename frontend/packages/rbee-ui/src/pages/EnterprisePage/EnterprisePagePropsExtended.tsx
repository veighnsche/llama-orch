// Extended Enterprise Page Props - Part 2
// This file contains the remaining props for Enterprise page sections

import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EnterpriseComparisonTemplateProps,
  EnterpriseComplianceTemplateProps,
  EnterpriseCTATemplateProps,
  EnterpriseFeaturesTemplateProps,
  EnterpriseHowItWorksTemplateProps,
  EnterpriseSecurityTemplateProps,
  EnterpriseSolutionTemplateProps,
  EnterpriseTestimonialsTemplateProps,
  EnterpriseUseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  Building2,
  Calendar,
  CheckCircle,
  Clock,
  Eye,
  FileCheck,
  FileText,
  Globe,
  Heart,
  KeyRound,
  Lock,
  MessageSquare,
  Rocket,
  Scale,
  Server,
  Shield,
  Users,
  Wrench,
} from 'lucide-react'
import Image from 'next/image'

// === Enterprise Solution ===

/**
 * Enterprise Solution container - wraps the solution section
 */
export const enterpriseSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'EU-Native AI Infrastructure That Meets Compliance by Design',
  description:
    'Enterprise-grade, self-hosted AI that keeps data sovereign, auditable, and under your control—EU resident, zero US cloud dependencies.',
  kicker: 'How rbee Works',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Solution - How rbee Works section
 */
export const enterpriseSolutionProps: EnterpriseSolutionTemplateProps = {
  id: 'how-it-works',
  kicker: 'How rbee Works',
  eyebrowIcon: <Shield className="h-4 w-4" aria-hidden="true" />,
  title: 'EU-Native AI Infrastructure That Meets Compliance by Design',
  subtitle:
    'Enterprise-grade, self-hosted AI that keeps data sovereign, auditable, and under your control—EU resident, zero US cloud dependencies.',
  features: [
    {
      icon: <Shield className="h-6 w-6" aria-hidden="true" />,
      title: '100% Data Sovereignty',
      body: 'Data stays on your infrastructure. EU-only deployment. Full control.',
      badge: 'GDPR Art. 44',
    },
    {
      icon: <Lock className="h-6 w-6" aria-hidden="true" />,
      title: '7-Year Audit Retention',
      body: 'Immutable, tamper-evident logs. Legally defensible.',
      badge: 'GDPR Art. 30',
    },
    {
      icon: <FileCheck className="h-6 w-6" aria-hidden="true" />,
      title: '32 Audit Event Types',
      body: 'Auth, data access, policy changes, compliance events.',
    },
    {
      icon: <Server className="h-6 w-6" aria-hidden="true" />,
      title: 'Zero US Cloud Dependencies',
      body: 'Self-hosted or EU marketplace. No Schrems II exposure.',
    },
  ],
  steps: [
    {
      title: 'Deploy On-Premises',
      body: 'Install rbee on your EU-based infrastructure. Full air-gap support.',
    },
    {
      title: 'Configure Compliance Policies',
      body: 'Set data residency rules, audit retention, and access controls via Rhai policies.',
    },
    {
      title: 'Enable Audit Logging',
      body: 'Immutable audit trail captures all authentication, data access, and compliance events.',
    },
    {
      title: 'Run Compliant AI',
      body: 'Your models, your data, your infrastructure. Zero external dependencies.',
    },
  ],
  earnings: {
    title: 'Compliance Metrics',
    rows: [
      {
        model: 'Data Sovereignty',
        meta: 'GDPR Art. 44',
        value: '100%',
        note: 'EU-only',
      },
      {
        model: 'Audit Retention',
        meta: 'GDPR Art. 30',
        value: '7 years',
        note: 'immutable',
      },
      {
        model: 'Security Layers',
        meta: 'Defense-in-depth',
        value: '5 layers',
        note: 'zero-trust',
      },
    ],
    disclaimer:
      'rbee is designed to meet GDPR, NIS2, and EU AI Act requirements. Consult your legal team for certification.',
  },
  illustration: (
    <Image
      src="/decor/eu-ledger-grid.webp"
      width={1200}
      height={640}
      className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
      alt="Abstract EU-blue ledger grid with softly glowing checkpoints, implying immutable audit trails and data sovereignty; premium dark UI, subtle amber accents"
      aria-hidden="true"
    />
  ),
  ctaPrimary: {
    label: 'Request Demo',
    href: '/enterprise/demo',
  },
  ctaSecondary: {
    label: 'View Compliance Docs',
    href: '/docs/compliance',
  },
  ctaCaption: 'EU data residency guaranteed; earnings/metrics depend on configuration.',
}

// === Enterprise Compliance ===

/**
 * Enterprise Compliance container - wraps the compliance section
 */
export const enterpriseComplianceContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Compliance by Design',
  description:
    'Built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements—security is engineered in, not bolted on.',
  kicker: 'Security & Certifications',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Compliance - Compliance by Design section
 */
export const enterpriseComplianceProps: EnterpriseComplianceTemplateProps = {
  id: 'compliance',
  backgroundImage: {
    src: '/decor/compliance-ledger.webp',
    alt: 'Abstract EU-blue ledger lines with checkpoint nodes; evokes immutable audit trails, GDPR alignment, SOC2 controls, ISO 27001 ISMS',
  },
  pillars: [
    {
      icon: Globe,
      title: 'GDPR',
      subtitle: 'EU Regulation',
      titleId: 'compliance-gdpr',
      bullets: [
        { title: '7-year audit retention (Art. 30)' },
        { title: 'Data access records (Art. 15)' },
        { title: 'Erasure tracking (Art. 17)' },
        { title: 'Consent management (Art. 7)' },
        { title: 'Data residency controls (Art. 44)' },
        { title: 'Breach notification (Art. 33)' },
      ],
      box: {
        heading: 'Compliance Endpoints',
        items: [
          'GET /v2/compliance/data-access',
          'POST /v2/compliance/data-export',
          'POST /v2/compliance/data-deletion',
          'GET /v2/compliance/audit-trail',
        ],
      },
    },
    {
      icon: Shield,
      title: 'SOC2',
      subtitle: 'US Standard',
      titleId: 'compliance-soc2',
      bullets: [
        { title: 'Auditor query API' },
        { title: '32 audit event types' },
        { title: '7-year retention (Type II)' },
        { title: 'Tamper-evident hash chains' },
        { title: 'Access control logging' },
        { title: 'Encryption at rest' },
      ],
      box: {
        heading: 'Trust Service Criteria',
        items: ['✓ Security (CC1-CC9)', '✓ Availability (A1.1-A1.3)', '✓ Confidentiality (C1.1-C1.2)'],
      },
    },
    {
      icon: Lock,
      title: 'ISO 27001',
      subtitle: 'International Standard',
      titleId: 'compliance-iso27001',
      bullets: [
        { title: 'Incident records (A.16)' },
        { title: '3-year minimum retention' },
        { title: 'Access logging (A.9)' },
        { title: 'Crypto controls (A.10)' },
        { title: 'Ops security (A.12)' },
        { title: 'Security policies (A.5)' },
      ],
      box: {
        heading: 'ISMS Controls',
        items: ['✓ 114 controls implemented', '✓ Risk assessment framework', '✓ Continuous monitoring'],
      },
    },
  ],
  auditReadiness: {
    heading: 'Ready for Your Compliance Audit',
    description: 'Download our compliance documentation package or schedule a call with our compliance team.',
    note: 'Pack includes endpoints, retention policy, and audit-logging design.',
    noteAriaLabel: 'Compliance pack includes endpoints, retention policy, and audit-logging design',
    buttons: [
      {
        text: 'Download Compliance Pack',
        href: '/compliance/download',
        ariaDescribedby: 'compliance-pack-note',
      },
      {
        text: 'Talk to Compliance Team',
        href: '/contact/compliance',
        variant: 'outline',
        ariaDescribedby: 'compliance-pack-note',
      },
    ],
    footnote: 'rbee (pronounced "are-bee")',
  },
}

// === Enterprise Security ===

/**
 * Enterprise Security container - wraps the security section
 */
export const enterpriseSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Enterprise-Grade Security',
  description:
    'Six specialized security crates harden every layer—from auth and inputs to secrets, auditing, JWT lifecycle, and time-bounded execution.',
  kicker: 'Defense-in-Depth',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Security - Defense-in-Depth section
 */
export const enterpriseSecurityProps: EnterpriseSecurityTemplateProps = {
  backgroundImage: {
    src: '/decor/security-mesh.webp',
    alt: 'Abstract dark security mesh with linked nodes and amber highlights, suggesting hash-chains, zero-trust, and time-bounded execution',
  },
  securityCrates: [
    {
      icon: Lock,
      title: 'auth-min: Zero-Trust Authentication',
      subtitle: 'The Trickster Guardians',
      intro:
        'Constant-time token checks stop CWE-208 leaks. Fingerprints let you log safely. Bind policies block accidental exposure.',
      bullets: [
        'Timing-safe comparison (constant-time)',
        'Token fingerprinting (SHA-256)',
        'Bearer token parsing (RFC 6750)',
        'Bind policy enforcement',
      ],
      docsHref: '/docs/security/auth-min',
    },
    {
      icon: Eye,
      title: 'audit-logging: Compliance Engine',
      subtitle: 'Legally Defensible Proof',
      intro: 'Append-only audit trail with 32 event types. Hash-chain tamper detection. 7-year retention for GDPR.',
      bullets: [
        'Immutable audit trail (append-only)',
        '32 event types across 7 categories',
        'Tamper detection (hash chains)',
        '7-year retention (GDPR)',
      ],
      docsHref: '/docs/security/audit-logging',
    },
    {
      icon: Shield,
      title: 'input-validation: First Line of Defense',
      subtitle: 'Trust No Input',
      intro: 'Prevents injection and exhaustion. Validates identifiers, prompts, paths—before execution.',
      bullets: [
        'SQL injection prevention',
        'Command injection prevention',
        'Path traversal prevention',
        'Resource exhaustion prevention',
      ],
      docsHref: '/docs/security/input-validation',
    },
    {
      icon: Server,
      title: 'secrets-management: Credential Guardian',
      subtitle: 'Never in Environment',
      intro: 'File-scoped secrets with zeroization and systemd credentials. Timing-safe verification.',
      bullets: [
        'File-based loading (not env vars)',
        'Memory zeroization on drop',
        'Permission validation (0600)',
        'Timing-safe verification',
      ],
      docsHref: '/docs/security/secrets-management',
    },
    {
      icon: KeyRound,
      title: 'jwt-guardian: Token Lifecycle Manager',
      subtitle: 'Stateless Yet Secure',
      intro: 'RS256 signature validation with clock-skew tolerance. Revocation lists and short-lived refresh tokens.',
      bullets: [
        'RS256/ES256 signature validation',
        'Clock-skew tolerance (±5 min)',
        'Revocation list (Redis-backed)',
        'Short-lived refresh tokens (15 min)',
      ],
      docsHref: '/docs/security/jwt-guardian',
    },
    {
      icon: Clock,
      title: 'deadline-propagation: Performance Enforcer',
      subtitle: 'Every Millisecond Counts',
      intro: 'Propagates time budgets end-to-end. Aborts doomed work to protect SLOs.',
      bullets: [
        'Deadline propagation (client → worker)',
        'Remaining time calculation',
        'Deadline enforcement (abort if insufficient)',
        'Timeout responses (504 Gateway Timeout)',
      ],
      docsHref: '/docs/security/deadline-propagation',
    },
  ],
  guarantees: {
    heading: 'Security Guarantees',
    stats: [
      {
        value: '< 10%',
        label: 'Timing variance (constant-time)',
        ariaLabel: 'Less than 10 percent timing variance',
      },
      {
        value: '100%',
        label: 'Token fingerprinting (no raw tokens)',
        ariaLabel: '100 percent token fingerprinting',
      },
      {
        value: 'Zero',
        label: 'Memory leaks (zeroization on drop)',
        ariaLabel: 'Zero memory leaks',
      },
    ],
    footnote: 'Figures represent default crate configurations; tune in policy for your environment.',
  },
}

// Continue in next file due to length...
