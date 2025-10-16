// Extended Enterprise Page Props - Part 3
// This file contains the final props for Enterprise page sections

import { TESTIMONIAL_STATS } from '@rbee/ui/data/testimonials'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { FEATURES, PROVIDERS } from '@rbee/ui/organisms/Enterprise/ComparisonData/comparison-data'
import type {
  EnterpriseComparisonTemplateProps,
  EnterpriseCTATemplateProps,
  EnterpriseFeaturesTemplateProps,
  EnterpriseHowItWorksTemplateProps,
  EnterpriseTestimonialsTemplateProps,
  EnterpriseUseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  Building2,
  Calendar,
  CheckCircle,
  FileText,
  Globe,
  Heart,
  MessageSquare,
  Rocket,
  Scale,
  Server,
  Shield,
  Users,
  Wrench,
} from 'lucide-react'

// === Enterprise How It Works ===

/**
 * Enterprise How It Works container - wraps the deployment process section
 */
export const enterpriseHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Enterprise Deployment Process',
  description: 'From consultation to production, we guide every step of your compliance journey.',
  kicker: 'Deployment & Compliance',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise How It Works - Deployment Process section
 */
export const enterpriseHowItWorksProps: EnterpriseHowItWorksTemplateProps = {
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

// === Enterprise Use Cases ===

/**
 * Enterprise Use Cases container - wraps the industry playbooks section
 */
export const enterpriseUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Built for Regulated Industries',
  description:
    'Organizations in high-compliance sectors run rbee on EU-resident infrastructure—no foreign clouds, audit-ready by design.',
  kicker: 'Industry Playbooks',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Use Cases - Industry Playbooks section
 */
export const enterpriseUseCasesProps: EnterpriseUseCasesTemplateProps = {
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
      { text: 'Request Industry Brief', href: '/contact/industry-brief' },
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

// === Enterprise Comparison ===

/**
 * Enterprise Comparison container - wraps the feature matrix section
 */
export const enterpriseComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Why Enterprises Choose rbee',
  description: "See how rbee's compliance and security compare to external AI providers.",
  kicker: 'Feature Matrix',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Comparison - Feature Matrix section
 */
export const enterpriseComparisonProps: EnterpriseComparisonTemplateProps = {
  providers: PROVIDERS,
  features: FEATURES,
  footnote: '* Comparison based on publicly available information as of October 2025.',
}

// === Enterprise Features ===

/**
 * Enterprise Features container - wraps the enterprise capabilities section
 */
export const enterpriseFeaturesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Enterprise Features',
  description: 'Everything you need for compliant, resilient, EU-resident AI infrastructure.',
  kicker: 'Enterprise Capabilities',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Features - Enterprise Capabilities section
 */
export const enterpriseFeaturesProps: EnterpriseFeaturesTemplateProps = {
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

// === Enterprise Testimonials ===

/**
 * Enterprise Testimonials container - wraps the testimonials section
 */
export const enterpriseTestimonialsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Trusted by Regulated Industries',
  description: 'Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Enterprise Testimonials - Trusted by Regulated Industries section
 */
export const enterpriseTestimonialsProps: EnterpriseTestimonialsTemplateProps = {
  sectorFilter: ['finance', 'healthcare', 'legal'],
  layout: 'grid',
  showStats: true,
}

// === Enterprise CTA ===

/**
 * Enterprise CTA - Final call to action section (no container, self-contained)
 */
export const enterpriseCTAProps: EnterpriseCTATemplateProps = {
  eyebrow: 'Get Audit-Ready',
  heading: 'Ready to Meet Your Compliance Requirements?',
  description: 'Book a demo with our compliance team, or download the documentation pack.',
  trustStats: TESTIMONIAL_STATS,
  ctaOptions: [
    {
      icon: <Calendar className="h-6 w-6" />,
      title: 'Schedule Demo',
      body: '30-minute demo with our compliance team. See rbee in action with live environment walkthrough.',
      tone: 'primary',
      eyebrow: 'Most Popular',
      note: '30-minute session • live environment',
      buttonText: 'Book Demo',
      buttonHref: '/enterprise/demo',
      buttonAriaLabel: 'Book a 30-minute demo',
    },
    {
      icon: <FileText className="h-6 w-6" />,
      title: 'Compliance Pack',
      body: 'Download GDPR, SOC2, and ISO 27001 documentation with audit-ready templates and checklists.',
      eyebrow: 'Self-Service',
      note: 'GDPR, SOC2, ISO 27001 summaries',
      buttonText: 'Download Docs',
      buttonHref: '/docs/compliance-pack',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Download compliance documentation pack',
    },
    {
      icon: <MessageSquare className="h-6 w-6" />,
      title: 'Talk to Sales',
      body: 'Discuss your specific compliance requirements and get a custom proposal tailored to your needs.',
      eyebrow: 'Custom Solutions',
      note: 'We respond within one business day',
      buttonText: 'Contact Sales',
      buttonHref: '/contact/sales',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Contact sales team',
    },
  ],
  footerCaption: 'Enterprise support 24/7 • Typical deployment: 6–8 weeks from consultation to production.',
}
