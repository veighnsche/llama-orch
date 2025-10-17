import { DeploymentFlow, EuLedgerGrid, SectorGrid, SecurityMesh } from '@rbee/ui/atoms'
import { FEATURES, PROVIDERS } from '@rbee/ui/data/enterprise-comparison'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  ComparisonTemplateProps,
  EmailCaptureProps,
  EnterpriseComplianceProps,
  EnterpriseCTAProps,
  EnterpriseHeroProps,
  EnterpriseHowItWorksProps,
  EnterpriseSecurityProps,
  EnterpriseUseCasesProps,
  FAQTemplateProps,
  ProblemTemplateProps,
  ProvidersEarningsGPUModel,
  ProvidersEarningsPreset,
  ProvidersEarningsProps,
  SolutionTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  Building2,
  Calendar,
  CheckCircle,
  Clock,
  Eye,
  FileCheck,
  FileText,
  FileX,
  Globe,
  Heart,
  KeyRound,
  Lock,
  MessageSquare,
  Rocket,
  Scale,
  Server,
  Shield,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Compliance Hero ===

export const complianceHeroProps: EnterpriseHeroProps = {
  badge: {
    icon: <Shield className="size-6" />,
    text: 'Compliance by Design',
  },
  heading: 'Meet GDPR, SOC2, and ISO 27001 Requirements Without Compromise',
  description:
    'Audit-ready AI infrastructure built for regulated industries. Immutable logs, EU data residency, and tamper-evident trails—compliance engineered in, not bolted on.',
  stats: [
    {
      value: '32',
      label: 'Audit Event Types',
      helpText: 'Comprehensive event tracking across auth, data, policy, and compliance categories',
    },
    {
      value: '7 Years',
      label: 'Retention Policy',
      helpText: 'GDPR Article 30 compliant audit log retention with tamper-evident hash chains',
    },
    {
      value: '100%',
      label: 'EU Data Residency',
      helpText: 'All data processed and stored within EU borders—zero US cloud dependencies',
    },
  ],
  primaryCta: {
    text: 'Download Compliance Pack',
    ariaLabel: 'Download compliance documentation pack',
  },
  secondaryCta: {
    text: 'Schedule Audit Demo',
    href: '#audit-demo',
  },
  helperText: 'GDPR, SOC2, and ISO 27001 aligned. Audit endpoints ready for your compliance team.',
  complianceChips: [
    {
      icon: <FileCheck className="h-3 w-3" />,
      label: 'GDPR Art. 30',
      ariaLabel: 'GDPR Article 30 compliant audit logging',
    },
    {
      icon: <Shield className="h-3 w-3" />,
      label: 'SOC2 Type II',
      ariaLabel: 'SOC2 Type II audit ready',
    },
    {
      icon: <Lock className="h-3 w-3" />,
      label: 'ISO 27001',
      ariaLabel: 'ISO 27001 aligned security controls',
    },
  ],
  auditConsole: {
    title: 'Live Audit Trail',
    badge: 'Tamper-Evident',
    filterButtons: [
      { label: 'All', ariaLabel: 'Filter: All events', active: true },
      { label: 'Auth', ariaLabel: 'Filter: Auth events' },
      { label: 'Data', ariaLabel: 'Filter: Data events' },
      { label: 'Policy', ariaLabel: 'Filter: Policy events' },
    ],
    events: [
      {
        event: 'compliance.audit_export',
        user: 'auditor@regulator.eu',
        time: '2025-10-17T09:15:42Z',
        displayTime: '2025-10-17 09:15:42 UTC',
        status: 'success',
      },
      {
        event: 'data.subject_access',
        user: 'dpo@company.eu',
        time: '2025-10-17T09:14:28Z',
        displayTime: '2025-10-17 09:14:28 UTC',
        status: 'success',
      },
      {
        event: 'policy.retention_update',
        user: 'admin@company.eu',
        time: '2025-10-17T09:12:15Z',
        displayTime: '2025-10-17 09:12:15 UTC',
        status: 'success',
      },
      {
        event: 'auth.mfa_success',
        user: 'compliance@company.eu',
        time: '2025-10-17T09:10:03Z',
        displayTime: '2025-10-17 09:10:03 UTC',
        status: 'success',
      },
    ],
    footer: {
      retention: 'Hash-chained • Immutable',
      tamperProof: '7-year retention',
    },
  },
  floatingBadges: [
    {
      label: 'Audit Events',
      value: '2.4M logged',
      ariaLabel: '2.4 million audit events logged',
      position: 'top-right',
    },
    {
      label: 'Compliance',
      value: '3 Frameworks',
      ariaLabel: 'Compliant with 3 regulatory frameworks',
      position: 'bottom-left',
    },
  ],
}

// === Email Capture ===

export const complianceEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Audit-Ready',
    showPulse: true,
  },
  headline: 'Get your compliance documentation pack',
  subheadline:
    'Download GDPR, SOC2, and ISO 27001 alignment guides, audit endpoints reference, and retention policy templates.',
  emailInput: {
    placeholder: 'dpo@yourcompany.eu',
    label: 'Email address',
  },
  submitButton: {
    label: 'Download Pack',
  },
  trustMessage: 'Used by regulated industries across finance, healthcare, and legal sectors.',
  successMessage: 'Check your inbox! Compliance pack sent.',
  communityFooter: {
    text: 'Need help with compliance?',
    linkText: 'Talk to Compliance Team',
    linkHref: '/contact/compliance',
    subtext: 'We respond within 1 business day',
  },
}

export const complianceEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}

// === Problem Template ===

export const complianceProblemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Hidden Cost of Non-Compliance',
  description:
    'Cloud AI providers create compliance gaps that put your organization at risk of fines, audits, and reputational damage.',
  kicker: 'Compliance Risks',
  kickerVariant: 'destructive',
  background: {
    variant: 'gradient-destructive',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
  ctaBanner: {
    copy: 'Eliminate compliance risks with EU-native AI infrastructure. Audit-ready from day one.',
    primary: { label: 'Download Compliance Pack', href: '/compliance/download' },
    secondary: { label: 'Schedule Audit Demo', href: '/compliance/demo' },
  },
}

export const complianceProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: <Globe className="h-6 w-6" />,
      title: 'Cross-Border Data Transfers',
      body: 'Your data flows to US servers. GDPR Article 44 violations. Schrems II compliance impossible. Data Protection Authorities issue warnings. Fines loom.',
      tone: 'destructive',
      tag: 'GDPR Art. 44',
    },
    {
      icon: <FileX className="h-6 w-6" />,
      title: 'No Audit Trail',
      body: 'Cannot prove data access. No immutable logs. GDPR Article 30 non-compliance. SOC2 audits fail. ISO 27001 certification blocked. Auditors reject your evidence.',
      tone: 'destructive',
      tag: 'Audit failure',
    },
    {
      icon: <Scale className="h-6 w-6" />,
      title: 'Regulatory Fines',
      body: 'GDPR: €20M or 4% revenue. HIPAA: $50K per record. PCI-DSS breaches destroy reputation. Legal costs spiral. Insurance premiums skyrocket.',
      tone: 'destructive',
      tag: 'Up to €20M',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'Failed Audits',
      body: 'SOC2 Type II rejected. ISO 27001 certification denied. Customer audits fail. Contracts cancelled. Revenue lost. Reputation damaged permanently.',
      tone: 'destructive',
      tag: 'Certification blocked',
    },
  ],
  gridClassName: 'md:grid-cols-2 lg:grid-cols-4',
}

// === Compliance Solution ===

export const complianceSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Compliance by Design: Audit-Ready AI Infrastructure',
  description:
    'Self-hosted AI with immutable audit trails, EU data residency, and tamper-evident logs—built to meet GDPR, SOC2, and ISO 27001 from the ground up.',
  kicker: 'The Solution',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[52rem] -translate-x-1/2 opacity-25 md:block">
        <EuLedgerGrid />
      </div>
    ),
  },
  ctas: {
    primary: {
      label: 'Download Compliance Pack',
      href: '/compliance/download',
      ariaLabel: 'Download compliance documentation pack',
    },
    secondary: {
      label: 'Schedule Audit Demo',
      href: '/compliance/demo',
      ariaLabel: 'Schedule a compliance audit demo',
    },
    caption: 'Audit endpoints ready. Compliance documentation included. Legal review recommended.',
  },
}

export const complianceSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Shield className="h-6 w-6" aria-hidden="true" />,
      title: 'Immutable Audit Logs',
      body: '32 event types. Hash-chained. Tamper-evident. 7-year retention.',
      badge: 'GDPR Art. 30',
    },
    {
      icon: <Globe className="h-6 w-6" aria-hidden="true" />,
      title: '100% EU Data Residency',
      body: 'Data never leaves EU. Zero US cloud dependencies. Full sovereignty.',
      badge: 'GDPR Art. 44',
    },
    {
      icon: <FileCheck className="h-6 w-6" aria-hidden="true" />,
      title: 'Audit Endpoints',
      body: 'Query API for auditors. Export compliance reports. Real-time access.',
    },
    {
      icon: <Lock className="h-6 w-6" aria-hidden="true" />,
      title: 'Defense-in-Depth Security',
      body: '6 security crates. Zero-trust architecture. Constant-time operations.',
    },
  ],
  steps: [
    {
      title: 'Deploy On-Premises',
      body: 'Install rbee on EU infrastructure. Air-gap support. Full control.',
    },
    {
      title: 'Configure Compliance Policies',
      body: 'Set retention rules, data residency, and audit event types via Rhai policies.',
    },
    {
      title: 'Enable Audit Logging',
      body: 'Activate immutable audit trail. Hash-chain verification. Tamper detection.',
    },
    {
      title: 'Pass Your Audit',
      body: 'Provide auditors with query API access. Export compliance reports. Certification ready.',
    },
  ],
  earnings: {
    title: 'Compliance Metrics',
    rows: [
      {
        model: 'Audit Events',
        meta: '7 categories',
        value: '32 types',
        note: 'comprehensive',
      },
      {
        model: 'Retention',
        meta: 'GDPR Art. 30',
        value: '7 years',
        note: 'immutable',
      },
      {
        model: 'Data Residency',
        meta: 'GDPR Art. 44',
        value: '100% EU',
        note: 'sovereign',
      },
    ],
    disclaimer:
      'rbee is designed to meet GDPR, SOC2, and ISO 27001 requirements. Final certification requires legal review and auditor approval.',
  },
}

// === Compliance Standards (EnterpriseCompliance) ===

export const complianceStandardsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Regulatory Frameworks',
  title: 'Built for GDPR, SOC2, and ISO 27001',
  description:
    'Three pillars of compliance—data protection, trust service criteria, and information security—engineered into every layer.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-6 w-[50rem] -translate-x-1/2 opacity-25">
        <EuLedgerGrid />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'compliance-standards',
  auditReadinessCTA: {
    heading: 'Ready for Your Next Audit',
    description:
      'Download our compliance pack or schedule a call with our compliance team to discuss your requirements.',
    note: 'Pack includes audit endpoints, retention policy, event types, and architecture diagrams.',
    noteAriaLabel: 'Compliance pack includes audit endpoints, retention policy, event types, and architecture diagrams',
    buttons: [
      {
        text: 'Download Compliance Pack',
        href: '/compliance/download',
        variant: 'default',
        ariaDescribedby: 'compliance-pack-note',
      },
      {
        text: 'Schedule Audit Demo',
        href: '/compliance/demo',
        variant: 'outline',
        ariaDescribedby: 'compliance-pack-note',
      },
    ],
    footnote: 'rbee (pronounced "are-bee")',
  },
}

export const complianceStandardsProps: EnterpriseComplianceProps = {
  pillars: [
    {
      icon: <Globe className="size-6" />,
      title: 'GDPR',
      subtitle: 'EU Data Protection',
      titleId: 'compliance-gdpr',
      bullets: [
        { title: 'Data subject rights (Art. 15-22)' },
        { title: 'Audit retention (Art. 30)' },
        { title: 'Data residency (Art. 44)' },
        { title: 'Breach notification (Art. 33)' },
        { title: 'Consent management (Art. 7)' },
        { title: 'Erasure tracking (Art. 17)' },
      ],
      box: {
        heading: 'GDPR Endpoints',
        items: [
          'GET /v2/compliance/data-access',
          'POST /v2/compliance/data-export',
          'POST /v2/compliance/data-deletion',
          'GET /v2/compliance/audit-trail',
        ],
        disabledCheckmarks: true,
      },
    },
    {
      icon: <Shield className="size-6" />,
      title: 'SOC2',
      subtitle: 'Trust Service Criteria',
      titleId: 'compliance-soc2',
      bullets: [
        { title: 'Auditor query API' },
        { title: '32 audit event types' },
        { title: 'Tamper-evident hash chains' },
        { title: '7-year retention (Type II)' },
        { title: 'Access control logging' },
        { title: 'Encryption at rest & in transit' },
      ],
      box: {
        heading: 'TSC Coverage',
        items: ['Security (CC1-CC9)', 'Availability (A1.1-A1.3)', 'Confidentiality (C1.1-C1.2)'],
        checkmarkColor: 'white',
      },
    },
    {
      icon: <Lock className="size-6" />,
      title: 'ISO 27001',
      subtitle: 'Information Security',
      titleId: 'compliance-iso27001',
      bullets: [
        { title: 'Incident records (A.16)' },
        { title: 'Access logging (A.9)' },
        { title: 'Crypto controls (A.10)' },
        { title: 'Ops security (A.12)' },
        { title: 'Security policies (A.5)' },
        { title: 'Risk assessment (A.6)' },
      ],
      box: {
        heading: 'ISMS Controls',
        items: ['114 controls implemented', 'Risk assessment framework', 'Continuous monitoring'],
        checkmarkColor: 'white',
      },
    },
  ],
}

// === Security Features (EnterpriseSecurity) ===

export const complianceSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Security Architecture',
  title: 'Six Security Crates for Defense-in-Depth',
  description: 'Every layer hardened—auth, audit, input validation, secrets, JWT lifecycle, and deadline enforcement.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[52rem] -translate-x-1/2 opacity-25 md:block">
        <SecurityMesh className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  securityGuarantees: {
    heading: 'Security Guarantees',
    stats: [
      {
        value: '< 10%',
        label: 'Timing variance (constant-time)',
        ariaLabel: 'Less than 10 percent timing variance',
      },
      {
        value: '100%',
        label: 'Token fingerprinting',
        ariaLabel: '100 percent token fingerprinting',
      },
      {
        value: 'Zero',
        label: 'Memory leaks (zeroization)',
        ariaLabel: 'Zero memory leaks',
      },
    ],
    footnote: 'Default configurations. Tune in policy for your environment.',
  },
}

export const complianceSecurityProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Lock className="size-6" />,
      title: 'auth-min: Zero-Trust Authentication',
      subtitle: 'Constant-Time Token Checks',
      intro:
        'Timing-safe token comparison stops CWE-208 leaks. Fingerprints enable safe logging. Bind policies prevent accidental exposure.',
      bullets: [
        'Constant-time comparison (CWE-208)',
        'Token fingerprinting (SHA-256)',
        'Bearer token parsing (RFC 6750)',
        'Bind policy enforcement',
      ],
      docsHref: '/docs/security/auth-min',
    },
    {
      icon: <Eye className="size-6" />,
      title: 'audit-logging: Compliance Engine',
      subtitle: 'Immutable Audit Trail',
      intro:
        'Append-only audit trail with 32 event types across 7 categories. Hash-chain tamper detection. 7-year retention for GDPR Article 30.',
      bullets: [
        'Immutable audit trail (append-only)',
        '32 event types (7 categories)',
        'Tamper detection (hash chains)',
        '7-year retention (GDPR Art. 30)',
      ],
      docsHref: '/docs/security/audit-logging',
    },
    {
      icon: <Shield className="size-6" />,
      title: 'input-validation: First Line of Defense',
      subtitle: 'Trust No Input',
      intro: 'Prevents SQL injection, command injection, path traversal, and resource exhaustion before execution.',
      bullets: [
        'SQL injection prevention',
        'Command injection prevention',
        'Path traversal prevention',
        'Resource exhaustion prevention',
      ],
      docsHref: '/docs/security/input-validation',
    },
    {
      icon: <Server className="size-6" />,
      title: 'secrets-management: Credential Guardian',
      subtitle: 'Never in Environment',
      intro:
        'File-scoped secrets with memory zeroization on drop. Systemd credentials support. Timing-safe verification.',
      bullets: [
        'File-based loading (not env vars)',
        'Memory zeroization on drop',
        'Permission validation (0600)',
        'Timing-safe verification',
      ],
      docsHref: '/docs/security/secrets-management',
    },
    {
      icon: <KeyRound className="size-6" />,
      title: 'jwt-guardian: Token Lifecycle Manager',
      subtitle: 'Stateless Yet Secure',
      intro:
        'RS256/ES256 signature validation with clock-skew tolerance. Revocation lists and short-lived refresh tokens.',
      bullets: [
        'RS256/ES256 signature validation',
        'Clock-skew tolerance (±5 min)',
        'Revocation list (Redis-backed)',
        'Short-lived refresh tokens (15 min)',
      ],
      docsHref: '/docs/security/jwt-guardian',
    },
    {
      icon: <Clock className="size-6" />,
      title: 'deadline-propagation: Performance Enforcer',
      subtitle: 'Every Millisecond Counts',
      intro: 'Propagates time budgets end-to-end from client to worker. Aborts doomed work to protect SLOs.',
      bullets: [
        'Deadline propagation (client → worker)',
        'Remaining time calculation',
        'Deadline enforcement (abort)',
        'Timeout responses (504)',
      ],
      docsHref: '/docs/security/deadline-propagation',
    },
  ],
}

// === Audit Process (EnterpriseHowItWorks) ===

export const complianceAuditProcessContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Audit Preparation',
  title: 'How to Prepare for Your Compliance Audit',
  description: 'Four steps from deployment to certification—guided by our compliance team.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[48rem] -translate-x-1/2 opacity-25 md:block">
        <DeploymentFlow className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const complianceAuditProcessProps: EnterpriseHowItWorksProps = {
  id: 'audit-process',
  deploymentSteps: [
    {
      index: 1,
      icon: <Shield className="h-6 w-6" />,
      title: 'Compliance Gap Analysis',
      intro:
        'We map your requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS) and identify gaps in your current setup.',
      items: ['Regulatory framework mapping', 'Data flow analysis', 'Risk assessment report', 'Compliance roadmap'],
    },
    {
      index: 2,
      icon: <Server className="h-6 w-6" />,
      title: 'Deploy & Configure',
      intro: 'Deploy rbee on EU infrastructure. Configure audit logging, retention policies, and data residency rules.',
      items: [
        'EU data center deployment',
        'Audit logging configuration',
        'Retention policy setup (7 years)',
        'Data residency enforcement',
      ],
    },
    {
      index: 3,
      icon: <CheckCircle className="h-6 w-6" />,
      title: 'Auditor Access',
      intro: 'Provide auditors with query API access. Export compliance reports. Review audit trail with your team.',
      items: [
        'Auditor credentials provisioning',
        'Compliance report exports',
        'Audit trail walkthrough',
        'Documentation package delivery',
      ],
    },
    {
      index: 4,
      icon: <Rocket className="h-6 w-6" />,
      title: 'Certification',
      intro:
        'Pass your audit. Receive SOC2 Type II, ISO 27001, or GDPR certification. Maintain compliance with quarterly reviews.',
      items: ['Audit completion', 'Certification issuance', 'Quarterly compliance reviews', 'Continuous monitoring'],
    },
  ],
  timeline: {
    heading: 'Typical Audit Timeline',
    description: 'From gap analysis to certification',
    weeks: [
      { week: 'Week 1-2', phase: 'Gap Analysis' },
      { week: 'Week 3-4', phase: 'Deployment & Configuration' },
      { week: 'Week 5-8', phase: 'Auditor Review' },
      { week: 'Week 9-10', phase: 'Certification' },
    ],
  },
}

// === Industry Use Cases (EnterpriseUseCases) ===

export const complianceUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Regulated Industries',
  title: 'Compliance Use Cases by Industry',
  description:
    'Organizations in finance, healthcare, legal, and government sectors run rbee for audit-ready AI infrastructure.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-6 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <SectorGrid className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctaRail: {
    heading: 'See how rbee fits your industry.',
    buttons: [
      {
        text: 'Download Industry Brief',
        href: '/compliance/industry-brief',
        variant: 'default',
      },
      {
        text: 'Talk to Compliance Team',
        href: '/contact/compliance',
        variant: 'outline',
      },
    ],
    links: [
      { text: 'Finance', href: '/industries/finance' },
      { text: 'Healthcare', href: '/industries/healthcare' },
      { text: 'Legal', href: '/industries/legal' },
      { text: 'Government', href: '/industries/government' },
    ],
  },
}

export const complianceUseCasesProps: EnterpriseUseCasesProps = {
  industryCases: [
    {
      icon: <Building2 className="size-6" />,
      industry: 'Financial Services',
      segments: 'Banks, Insurance, FinTech',
      badges: ['PCI-DSS', 'GDPR', 'SOC2'],
      summary: 'EU bank needed AI for fraud detection but PCI-DSS and GDPR blocked external providers.',
      challenges: [
        'PCI-DSS: No external APIs',
        'GDPR: EU data residency',
        'SOC2: Complete audit trail',
        '7-year retention required',
      ],
      solutions: [
        'On-prem deployment (EU DC)',
        'Immutable audit logs (7-year)',
        'Zero external dependencies',
        'SOC2 Type II ready',
      ],
      href: '/industries/finance',
    },
    {
      icon: <Heart className="size-6" />,
      industry: 'Healthcare',
      segments: 'Hospitals, MedTech, Pharma',
      badges: ['HIPAA', 'GDPR Art. 9'],
      summary: 'Hospital needed AI-assisted diagnostics with HIPAA and GDPR Article 9 compliance.',
      challenges: [
        'HIPAA: PHI protection',
        'GDPR Art. 9: Health data',
        'No US cloud providers',
        'Breach notification required',
      ],
      solutions: [
        'Self-hosted (hospital DC)',
        'EU-only deployment',
        'Full audit trail (breach detection)',
        'HIPAA-aligned architecture',
      ],
      href: '/industries/healthcare',
    },
    {
      icon: <Scale className="size-6" />,
      industry: 'Legal Services',
      segments: 'Law Firms, LegalTech',
      badges: ['GDPR', 'Legal Hold'],
      summary: 'Law firm needed document analysis without risking attorney-client privilege.',
      challenges: [
        'Attorney-client privilege',
        'No external uploads',
        'Legal-hold audit trail',
        'EU residency required',
      ],
      solutions: ['On-prem (firm servers)', 'Zero data transfer', 'Immutable legal-hold logs', 'Full confidentiality'],
      href: '/industries/legal',
    },
    {
      icon: <Shield className="size-6" />,
      industry: 'Government',
      segments: 'Public Sector, Defense',
      badges: ['ISO 27001', 'Sovereignty'],
      summary: 'Government agency needed citizen services with strict sovereignty and security controls.',
      challenges: ['Data sovereignty', 'No foreign clouds', 'Transparent audit trail', 'ISO 27001 required'],
      solutions: ['Gov DC deployment', 'EU-only infra', 'ISO 27001 aligned', 'Complete sovereignty'],
      href: '/industries/government',
    },
  ],
}

// === Comparison Template ===

export const complianceComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Feature Matrix',
  title: 'Compliance Comparison: rbee vs Cloud AI',
  description: 'See how rbee stacks up against cloud AI providers on compliance and security features.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  footerCTA: {
    message: '* Comparison based on publicly available information as of October 2025.',
  },
}

export const complianceComparisonProps: ComparisonTemplateProps = {
  columns: PROVIDERS,
  rows: FEATURES,
  showMobileCards: true,
}

// === Audit Cost Calculator (ProvidersEarnings adapted) ===

export const complianceAuditCostCalculatorContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Cost Estimator',
  title: 'Calculate Your Audit Costs',
  description:
    'Estimate the cost of storing and managing audit logs for compliance. Based on event volume, retention period, and storage requirements.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

const auditEventTypes: ProvidersEarningsGPUModel[] = [
  { name: 'Small (< 10K events/day)', baseRate: 0.05, vram: 10 },
  { name: 'Medium (10K-100K events/day)', baseRate: 0.15, vram: 50 },
  { name: 'Large (100K-1M events/day)', baseRate: 0.5, vram: 500 },
  { name: 'Enterprise (> 1M events/day)', baseRate: 2.0, vram: 5000 },
]

const auditPresets: ProvidersEarningsPreset[] = [
  { label: 'Startup', hours: 7, utilization: 50 },
  { label: 'Growth', hours: 7, utilization: 75 },
  { label: 'Enterprise', hours: 7, utilization: 100 },
]

export const complianceAuditCostCalculatorProps: ProvidersEarningsProps = {
  gpuModels: auditEventTypes,
  presets: auditPresets,
  commission: 0,
  configTitle: 'Audit Configuration',
  selectGPULabel: 'Select Event Volume',
  presetsLabel: 'Quick Presets',
  hoursLabel: 'Retention (years)',
  utilizationLabel: 'Storage Efficiency (%)',
  earningsTitle: 'Estimated Audit Costs',
  monthlyLabel: 'Monthly Storage Cost',
  basedOnText: (years: number, efficiency: number) => `Based on ${years}-year retention at ${efficiency}% efficiency`,
  takeHomeLabel: 'Total Annual Cost',
  dailyLabel: 'Daily Cost',
  yearlyLabel: 'Annual Cost',
  breakdownTitle: 'Cost Breakdown',
  hourlyRateLabel: 'Storage Rate',
  hoursPerMonthLabel: 'Retention Period',
  utilizationBreakdownLabel: 'Storage Efficiency',
  commissionLabel: 'Overhead',
  yourTakeHomeLabel: 'Total Cost',
  ctaLabel: 'Download Compliance Pack',
  ctaAriaLabel: 'Download compliance documentation pack',
  secondaryCTALabel: 'View Event Types',
  formatCurrency: (n: number, opts?: Intl.NumberFormatOptions) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
      ...opts,
    }).format(n),
  formatHourly: (n: number) => `€${n.toFixed(2)}/day`,
}

// === FAQ Template ===

export const complianceFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Common Questions',
  title: 'Compliance FAQs',
  description: 'Answers to frequently asked questions about rbee compliance and audit readiness.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
}

export const complianceFAQProps: FAQTemplateProps = {
  categories: ['General', 'GDPR', 'SOC2', 'ISO 27001', 'Technical'],
  faqItems: [
    {
      value: 'gdpr-compliant',
      question: 'Is rbee GDPR compliant?',
      answer:
        'Yes. rbee is designed to meet GDPR requirements including Article 30 (audit retention), Article 44 (data residency), and Articles 15-22 (data subject rights). We provide audit endpoints for data access, export, and deletion requests.',
      category: 'GDPR',
    },
    {
      value: 'audit-events',
      question: 'What audit event types does rbee track?',
      answer:
        'rbee tracks 32 audit event types across 7 categories: authentication, data access, policy changes, compliance events, security incidents, system operations, and user actions. All events are immutably logged with tamper-evident hash chains.',
      category: 'Technical',
    },
    {
      value: 'retention',
      question: 'How long are audit logs retained?',
      answer:
        'Audit logs are retained for 7 years by default to meet GDPR Article 30 requirements. Retention policies are configurable via Rhai policies to meet your specific regulatory requirements (e.g., HIPAA, PCI-DSS, ISO 27001).',
      category: 'GDPR',
    },
    {
      value: 'auditor-access',
      question: 'Can auditors access the audit trail?',
      answer:
        'Yes. rbee provides a query API for auditors with read-only access to the audit trail. Auditors can export compliance reports, search events by type/user/date, and verify hash-chain integrity. Access is logged and requires MFA.',
      category: 'General',
    },
    {
      value: 'data-storage',
      question: 'Where is data stored?',
      answer:
        'All data is stored within the EU. rbee supports deployment in EU data centers (Frankfurt, Amsterdam, Paris) or on your own EU-based infrastructure. Zero dependencies on US cloud providers. Full data sovereignty guaranteed.',
      category: 'GDPR',
    },
    {
      value: 'soc2-ready',
      question: 'Is rbee SOC2 Type II ready?',
      answer:
        'Yes. rbee implements Trust Service Criteria for Security (CC1-CC9), Availability (A1.1-A1.3), and Confidentiality (C1.1-C1.2). We provide auditor access, tamper-evident logs, and 7-year retention for SOC2 Type II audits.',
      category: 'SOC2',
    },
    {
      value: 'iso27001-support',
      question: 'Does rbee support ISO 27001?',
      answer:
        'Yes. rbee aligns with ISO 27001 controls including incident records (A.16), access logging (A.9), crypto controls (A.10), ops security (A.12), security policies (A.5), and risk assessment (A.6). 114 controls implemented.',
      category: 'ISO 27001',
    },
    {
      value: 'custom-events',
      question: 'Can I customize audit event types?',
      answer:
        'Yes. You can define custom audit event types via Rhai policies. Custom events are logged with the same immutability and tamper-evident guarantees as built-in events. Useful for industry-specific compliance requirements.',
      category: 'Technical',
    },
    {
      value: 'tamper-detection',
      question: 'What happens if the audit log is tampered with?',
      answer:
        'Audit logs use hash-chain verification. Each event is cryptographically linked to the previous event. Any tampering breaks the chain and is immediately detected. Verification can be performed by auditors or automated monitoring.',
      category: 'Technical',
    },
    {
      value: 'getting-started',
      question: 'How do I get started with compliance?',
      answer:
        'Download our compliance pack for GDPR, SOC2, and ISO 27001 alignment guides. Schedule a call with our compliance team to discuss your requirements. We provide gap analysis, deployment support, and auditor access setup.',
      category: 'General',
    },
  ],
}

// === Compliance CTA ===

export const complianceCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: <span className="font-sans text-orange-500">Get Audit-Ready</span>,
  title: 'Ready to Pass Your Next Audit?',
  description: 'Download our compliance pack or schedule a demo with our compliance team.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
  ctas: {
    caption: 'Compliance support 24/7 • Typical audit preparation: 8–10 weeks from gap analysis to certification.',
  },
}

export const complianceCTAProps: EnterpriseCTAProps = {
  trustStats: [
    { value: '32', label: 'Audit Event Types' },
    { value: '7 Years', label: 'Retention Policy' },
    { value: '100%', label: 'EU Data Residency' },
  ],
  ctaOptions: [
    {
      icon: <FileText className="h-6 w-6" />,
      title: 'Download Compliance Pack',
      body: 'GDPR, SOC2, and ISO 27001 alignment guides, audit endpoints reference, and retention policy templates.',
      tone: 'primary',
      eyebrow: 'Most Popular',
      note: 'Instant download • PDF format',
      buttonText: 'Download Pack',
      buttonHref: '/compliance/download',
      buttonAriaLabel: 'Download compliance documentation pack',
    },
    {
      icon: <Calendar className="h-6 w-6" />,
      title: 'Schedule Audit Demo',
      body: '30-minute demo with our compliance team. See audit trail, query API, and compliance reports in action.',
      eyebrow: 'Live Demo',
      note: '30-minute session • live environment',
      buttonText: 'Book Demo',
      buttonHref: '/compliance/demo',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Book a 30-minute audit demo',
    },
    {
      icon: <MessageSquare className="h-6 w-6" />,
      title: 'Talk to Compliance Team',
      body: 'Discuss your specific compliance requirements and get a custom gap analysis and roadmap.',
      eyebrow: 'Custom Solutions',
      note: 'We respond within one business day',
      buttonText: 'Contact Team',
      buttonHref: '/contact/compliance',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Contact compliance team',
    },
  ],
}
