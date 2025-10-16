import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EmailCaptureProps,
  EnterpriseComparisonTemplateProps,
  EnterpriseComplianceTemplateProps,
  EnterpriseFeaturesTemplateProps,
  EnterpriseHowItWorksTemplateProps,
  EnterpriseSecurityTemplateProps,
  EnterpriseTestimonialsTemplateProps,
  EnterpriseUseCasesTemplateProps,
  ProblemTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  Building2,
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
  Rocket,
  Scale,
  Server,
  Shield,
  Users,
  Wrench,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Email Capture ===

export const enterpriseEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'For Enterprise',
    showPulse: false,
  },
  headline: 'GDPR-compliant AI infrastructure',
  subheadline: 'Join enterprises that have taken control of their compliance and data sovereignty.',
  emailInput: {
    placeholder: 'your@company.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Request Demo',
  },
  trustMessage: 'SOC2 Type II certified. ISO 27001 compliant.',
  successMessage: 'Thanks! Our team will contact you within 24 hours.',
  communityFooter: {
    text: 'Read our compliance documentation',
    linkText: 'Compliance Overview',
    linkHref: '/enterprise/compliance',
    subtext: 'Download our security whitepaper',
  },
}

// === Problem Template ===

export const enterpriseProblemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Compliance Challenge of Cloud AI',
  description:
    'Using external AI providers creates compliance risks that can cost millions in fines and damage your reputation.',
  kicker: 'The Compliance Risk',
  kickerVariant: 'destructive',
  bgVariant: 'destructive-gradient',
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: <Globe className="h-6 w-6" />,
      title: 'Data Sovereignty Violations',
      body: 'Your sensitive data crosses borders to US cloud providers. GDPR Article 44 violations. Schrems II compliance impossible. Data Protection Authorities watching.',
      tone: 'destructive',
      tag: 'GDPR Art. 44',
    },
    {
      icon: <FileX className="h-6 w-6" />,
      title: 'Missing Audit Trails',
      body: 'No immutable logs. No proof of compliance. Cannot demonstrate GDPR Article 30 compliance. SOC2 audits fail. ISO 27001 certification impossible.',
      tone: 'destructive',
      tag: 'Audit failure',
    },
    {
      icon: <Scale className="h-6 w-6" />,
      title: 'Regulatory Fines',
      body: 'GDPR fines up to €20M or 4% of global revenue. Healthcare (HIPAA) violations: $50K per record. Financial services (PCI-DSS) breaches: reputation destroyed.',
      tone: 'destructive',
      tag: 'Up to €20M',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'Zero Control',
      body: 'Provider changes terms. Data Processing Agreements worthless. Cannot guarantee data residency. Cannot prove compliance. Your DPO cannot sleep.',
      tone: 'destructive',
      tag: 'No guarantees',
    },
  ],
  ctaPrimary: { label: 'Request Demo', href: '/enterprise/demo' },
  ctaSecondary: {
    label: 'Compliance Overview',
    href: '/enterprise/compliance',
  },
  ctaCopy:
    '"We cannot use external AI providers due to GDPR compliance requirements." — Every EU CTO and Data Protection Officer',
  gridClassName: 'md:grid-cols-2 lg:grid-cols-4',
}

// === EnterpriseCompliance Template ===

export const enterpriseComplianceContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Security & Certifications',
  title: 'Compliance by Design',
  description:
    'Built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements—security is engineered in, not bolted on.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseComplianceProps: Omit<EnterpriseComplianceTemplateProps, 'eyebrow' | 'heading' | 'description'> =
  {
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
          variant: 'default',
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

// === EnterpriseSecurity Template ===

export const enterpriseSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Defense-in-Depth',
  title: 'Enterprise-Grade Security',
  description:
    'Six specialized security crates harden every layer—from auth and inputs to secrets, auditing, JWT lifecycle, and time-bounded execution.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseSecurityProps: Omit<EnterpriseSecurityTemplateProps, 'eyebrow' | 'heading' | 'description'> = {
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

// === EnterpriseHowItWorks Template ===

export const enterpriseHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Deployment & Compliance',
  title: 'Enterprise Deployment Process',
  description: 'From consultation to production, we guide every step of your compliance journey.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseHowItWorksProps: Omit<EnterpriseHowItWorksTemplateProps, 'eyebrow' | 'heading' | 'description'> =
  {
    id: 'deployment',
    backgroundImage: {
      src: '/decor/deployment-flow.webp',
      alt: 'Abstract EU-blue flow diagram with four checkpoints and connecting lines, suggesting enterprise deployment stages and compliance handoffs',
    },
    deploymentSteps: [
      {
        index: 1,
        icon: <Shield className="h-6 w-6" />,
        title: 'Compliance Assessment',
        intro:
          'We map requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS), define residency, retention, and security controls.',
        items: [
          'Compliance gap analysis',
          'Data flow mapping',
          'Risk assessment report',
          'Deployment architecture proposal',
        ],
      },
      {
        index: 2,
        icon: <Server className="h-6 w-6" />,
        title: 'On-Premises Deployment',
        intro:
          'Deploy in EU data centers or on your servers. Configure EU-only workers, audit logging, and controls. White-label optional.',
        items: [
          'EU data centers (Frankfurt, Amsterdam, Paris)',
          'On-premises (your servers)',
          'Private cloud (AWS EU, Azure EU, GCP EU)',
          'Hybrid (on-prem + marketplace)',
        ],
      },
      {
        index: 3,
        icon: <CheckCircle className="h-6 w-6" />,
        title: 'Compliance Validation',
        intro:
          'Work with auditors: provide audit-trail access, docs, and architecture reviews. Supports SOC2 Type II, ISO 27001, GDPR.',
        items: [
          'Compliance documentation package',
          'Auditor access to audit logs',
          'Security architecture review',
          'Penetration testing reports',
        ],
      },
      {
        index: 4,
        icon: <Rocket className="h-6 w-6" />,
        title: 'Production Launch',
        intro: 'Go live with enterprise SLAs, 24/7 support, monitoring, and compliance reporting. Scale as you grow.',
        items: [
          '99.9% uptime SLA',
          '24/7 support (1-hour response time)',
          'Dedicated account manager',
          'Quarterly compliance reviews',
        ],
      },
    ],
    timeline: {
      heading: 'Typical Deployment Timeline',
      description: 'From consultation to production',
      weeks: [
        { week: 'Week 1-2', phase: 'Compliance Assessment' },
        { week: 'Week 3-4', phase: 'Deployment & Configuration' },
        { week: 'Week 5-6', phase: 'Compliance Validation' },
        { week: 'Week 7', phase: 'Production Launch' },
      ],
    },
  }

// === EnterpriseUseCases Template ===

export const enterpriseUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Industry Playbooks',
  title: 'Built for Regulated Industries',
  description:
    'Organizations in high-compliance sectors run rbee on EU-resident infrastructure—no foreign clouds, audit-ready by design.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseUseCasesProps: Omit<EnterpriseUseCasesTemplateProps, 'eyebrow' | 'heading' | 'description'> = {
  backgroundImage: {
    src: '/decor/sector-grid.webp',
    alt: 'Abstract EU-blue grid of industry tiles—finance, healthcare, legal, government—with soft amber accents; premium dark UI, compliance theme',
  },
  industryCases: [
    {
      icon: Building2,
      industry: 'Financial Services',
      segments: 'Banks, Insurance, FinTech',
      badges: ['PCI-DSS', 'GDPR', 'SOC2'],
      summary: 'EU bank needed internal code-gen but PCI-DSS/GDPR blocked external AI.',
      challenges: [
        'No external APIs (PCI-DSS)',
        'Complete audit trail (SOC2)',
        'EU data residency (GDPR)',
        '7-year retention',
      ],
      solutions: [
        'On-prem (EU data center)',
        'Immutable audit logs (7-year)',
        'Zero external dependencies',
        'SOC2 Type II ready',
      ],
      href: '/industries/finance',
    },
    {
      icon: Heart,
      industry: 'Healthcare',
      segments: 'Hospitals, MedTech, Pharma',
      badges: ['HIPAA', 'GDPR Art. 9'],
      summary: 'AI-assisted patient tooling with HIPAA + GDPR Article 9 constraints.',
      challenges: ['HIPAA/PHI protection', 'GDPR Art. 9 (health data)', 'No US clouds', 'Breach notifications'],
      solutions: [
        'Self-hosted (hospital DC)',
        'EU-only deployment',
        'Full audit trail (breach detection)',
        'HIPAA-aligned architecture',
      ],
      href: '/industries/healthcare',
    },
    {
      icon: Scale,
      industry: 'Legal Services',
      segments: 'Law Firms, LegalTech',
      badges: ['GDPR', 'Legal Hold'],
      summary: 'Document analysis without risking privilege.',
      challenges: ['Attorney-client privilege', 'No external uploads', 'Legal-hold audit trail', 'EU residency'],
      solutions: ['On-prem (firm servers)', 'Zero data transfer', 'Immutable legal-hold logs', 'Full confidentiality'],
      href: '/industries/legal',
    },
    {
      icon: Shield,
      industry: 'Government',
      segments: 'Public Sector, Defense',
      badges: ['ISO 27001', 'Sovereignty'],
      summary: 'Citizen services with strict sovereignty + security controls.',
      challenges: ['Data sovereignty', 'No foreign clouds', 'Transparent audit trail', 'ISO 27001 required'],
      solutions: ['Gov DC deployment', 'EU-only infra', 'ISO 27001 aligned', 'Complete sovereignty'],
      href: '/industries/government',
    },
  ],
  cta: {
    text: 'See how rbee fits your sector.',
    buttons: [
      {
        text: 'Request Industry Brief',
        href: '/contact/industry-brief',
        variant: 'default',
      },
      {
        text: 'Talk to a Solutions Architect',
        href: '/contact/solutions',
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

// === EnterpriseComparison Template ===
// Note: Import PROVIDERS and FEATURES from comparison-data.ts

export const enterpriseComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Feature Matrix',
  title: 'Why Enterprises Choose rbee',
  description: "See how rbee's compliance and security compare to external AI providers.",
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

// This will need to import PROVIDERS and FEATURES from comparison-data.ts
export const enterpriseComparisonProps: Omit<
  EnterpriseComparisonTemplateProps,
  'eyebrow' | 'heading' | 'description' | 'disclaimer'
> = {
  providers: [], // Import from comparison-data.ts: PROVIDERS
  features: [], // Import from comparison-data.ts: FEATURES
  footnote: '* Comparison based on publicly available information as of October 2025.',
}

// === EnterpriseFeatures Template ===

export const enterpriseFeaturesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Enterprise Capabilities',
  title: 'Enterprise Features',
  description: 'Everything you need for compliant, resilient, EU-resident AI infrastructure.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseFeaturesProps: Omit<EnterpriseFeaturesTemplateProps, 'eyebrow' | 'heading' | 'description'> = {
  features: [
    {
      icon: <Shield />,
      title: 'Enterprise SLAs',
      intro: '99.9% uptime with 24/7 support and 1-hour response. Dedicated manager and quarterly reviews.',
      bullets: ['99.9% SLA', '24/7 support (1-hour)', 'Dedicated account manager', 'Quarterly reviews'],
    },
    {
      icon: <Users />,
      title: 'White-Label Option',
      intro: 'Run rbee as your brand—custom domain, UI, and endpoints.',
      bullets: [
        'Custom branding/logo',
        'Custom domain (ai.yourcompany.com)',
        'UI customization',
        'API endpoint customization',
      ],
    },
    {
      icon: <Wrench />,
      title: 'Professional Services',
      intro: 'Deployment, integration, optimization, and training from our team.',
      bullets: ['Deployment consulting', 'Integration support', 'Custom development', 'Team training'],
    },
    {
      icon: <Globe />,
      title: 'Multi-Region Support',
      intro: 'EU multi-region for redundancy and compliance: failover + load balancing.',
      bullets: ['EU multi-region', 'Automatic failover', 'Load balancing', 'Geo-redundancy'],
    },
  ],
  outcomes: {
    heading: 'What you get',
    stats: [
      { value: '99.9%', label: 'Uptime SLA' },
      { value: '< 1 hr', label: 'Support response' },
      { value: 'EU-only', label: 'Data residency' },
    ],
    linkText: 'See compliance details',
    linkHref: '#compliance',
  },
}

// === EnterpriseTestimonials Template ===

export const enterpriseTestimonialsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Trusted by Regulated Industries',
  description: 'Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const enterpriseTestimonialsProps: Omit<EnterpriseTestimonialsTemplateProps, 'heading' | 'description'> = {
  sectorFilter: ['finance', 'healthcare', 'legal'],
  layout: 'grid',
  showStats: true,
}
