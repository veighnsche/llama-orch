import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EmailCaptureProps,
  EnterpriseComparisonTemplateProps,
  EnterpriseComplianceTemplateProps,
  EnterpriseCTATemplateProps,
  EnterpriseFeaturesTemplateProps,
  EnterpriseHeroTemplateProps,
  EnterpriseHowItWorksTemplateProps,
  EnterpriseSecurityTemplateProps,
  EnterpriseSolutionTemplateProps,
  EnterpriseTestimonialsTemplateProps,
  EnterpriseUseCasesTemplateProps,
  ProblemTemplateProps,
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
  Filter,
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

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Enterprise Hero ===

/**
 * Enterprise Hero - Main hero section with audit console visual
 */
export const enterpriseHeroProps: EnterpriseHeroTemplateProps = {
  badge: {
    icon: Shield,
    text: 'EU-Native AI Infrastructure',
  },
  heading: 'AI Infrastructure That Meets Your Compliance Requirements',
  description:
    'GDPR-compliant by design. SOC2 ready. ISO 27001 aligned. Build AI on your terms with EU data residency, immutable audit trails, and enterprise-grade security.',
  stats: [
    {
      value: '100%',
      label: 'GDPR Compliant',
      helpText: 'Full compliance with EU General Data Protection Regulation',
    },
    {
      value: '7 Years',
      label: 'Audit Retention',
      helpText: 'Immutable audit logs retained for 7 years per GDPR requirements',
    },
    {
      value: 'Zero',
      label: 'US Cloud Deps',
      helpText: 'No dependencies on US cloud providers; EU-native infrastructure',
    },
  ],
  primaryCta: {
    text: 'Schedule Demo',
    ariaLabel: 'Schedule a compliance demo',
  },
  secondaryCta: {
    text: 'View Compliance Details',
    href: '#compliance',
  },
  helperText: 'EU data residency guaranteed. Audited event types updated quarterly.',
  complianceChips: [
    {
      icon: <FileCheck className="h-3 w-3" />,
      label: 'GDPR Compliant',
      ariaLabel: 'GDPR Compliant certification',
    },
    {
      icon: <Shield className="h-3 w-3" />,
      label: 'SOC2 Ready',
      ariaLabel: 'SOC2 Ready certification',
    },
    {
      icon: <Lock className="h-3 w-3" />,
      label: 'ISO 27001 Aligned',
      ariaLabel: 'ISO 27001 Aligned certification',
    },
  ],
  auditConsole: {
    title: 'Immutable Audit Trail',
    badge: 'Compliant',
    filterButtons: [
      { label: 'All', ariaLabel: 'Filter: All events', active: true },
      { label: 'Auth', ariaLabel: 'Filter: Auth events' },
      { label: 'Data', ariaLabel: 'Filter: Data events' },
      { label: 'Exports', ariaLabel: 'Filter: Export events' },
    ],
    events: [
      {
        event: 'auth.success',
        user: 'admin@company.eu',
        time: '2025-10-11T14:23:15Z',
        displayTime: '2025-10-11 14:23:15 UTC',
        status: 'success',
      },
      {
        event: 'data.access',
        user: 'analyst@company.eu',
        time: '2025-10-11T14:22:48Z',
        displayTime: '2025-10-11 14:22:48 UTC',
        status: 'success',
      },
      {
        event: 'task.submitted',
        user: 'dev@company.eu',
        time: '2025-10-11T14:21:33Z',
        displayTime: '2025-10-11 14:21:33 UTC',
        status: 'success',
      },
      {
        event: 'compliance.export',
        user: 'dpo@company.eu',
        time: '2025-10-11T14:20:12Z',
        displayTime: '2025-10-11 14:20:12 UTC',
        status: 'success',
      },
    ],
    footer: {
      retention: 'Retention: 7 years (GDPR)',
      tamperProof: 'Tamper-evident',
    },
  },
  floatingBadges: [
    {
      label: 'Data Residency',
      value: '🇪🇺 EU Only',
      ariaLabel: 'All data processed and stored within the EU',
      position: 'top-right',
    },
    {
      label: 'Audit Events',
      value: '32 Types',
      ariaLabel: '32 distinct audit event types tracked',
      position: 'bottom-left',
    },
  ],
}

// === Email Capture ===

/**
 * Email Capture - Enterprise-focused messaging
 */
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

/**
 * Problem template container - wraps the problem cards section
 */
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

/**
 * Problem template data - The Compliance Challenge of Cloud AI
 */
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
