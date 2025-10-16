import type { EmailCaptureProps, ProblemTemplateProps } from "@rbee/ui/templates";
import type { TemplateContainerProps } from "@rbee/ui/molecules";
import { AlertTriangle, FileX, Globe, Scale } from "lucide-react";

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Email Capture ===

/**
 * Email Capture - Enterprise-focused messaging
 */
export const enterpriseEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: "For Enterprise",
    showPulse: false,
  },
  headline: "GDPR-compliant AI infrastructure",
  subheadline: "Join enterprises that have taken control of their compliance and data sovereignty.",
  emailInput: {
    placeholder: "your@company.com",
    label: "Email address",
  },
  submitButton: {
    label: "Request Demo",
  },
  trustMessage: "SOC2 Type II certified. ISO 27001 compliant.",
  successMessage: "Thanks! Our team will contact you within 24 hours.",
  communityFooter: {
    text: "Read our compliance documentation",
    linkText: "Compliance Overview",
    linkHref: "/enterprise/compliance",
    subtext: "Download our security whitepaper",
  },
};

// === Problem Template ===

/**
 * Problem template container - wraps the problem cards section
 */
export const enterpriseProblemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: "The Compliance Challenge of Cloud AI",
  description: "Using external AI providers creates compliance risks that can cost millions in fines and damage your reputation.",
  kicker: "The Compliance Risk",
  kickerVariant: "destructive",
  bgVariant: "destructive-gradient",
  paddingY: "xl",
  maxWidth: "7xl",
  align: "center",
}

/**
 * Problem template data - The Compliance Challenge of Cloud AI
 */
export const enterpriseProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: <Globe className="h-6 w-6" />,
      title: "Data Sovereignty Violations",
      body: "Your sensitive data crosses borders to US cloud providers. GDPR Article 44 violations. Schrems II compliance impossible. Data Protection Authorities watching.",
      tone: "destructive",
      tag: "GDPR Art. 44",
    },
    {
      icon: <FileX className="h-6 w-6" />,
      title: "Missing Audit Trails",
      body: "No immutable logs. No proof of compliance. Cannot demonstrate GDPR Article 30 compliance. SOC2 audits fail. ISO 27001 certification impossible.",
      tone: "destructive",
      tag: "Audit failure",
    },
    {
      icon: <Scale className="h-6 w-6" />,
      title: "Regulatory Fines",
      body: "GDPR fines up to €20M or 4% of global revenue. Healthcare (HIPAA) violations: $50K per record. Financial services (PCI-DSS) breaches: reputation destroyed.",
      tone: "destructive",
      tag: "Up to €20M",
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: "Zero Control",
      body: "Provider changes terms. Data Processing Agreements worthless. Cannot guarantee data residency. Cannot prove compliance. Your DPO cannot sleep.",
      tone: "destructive",
      tag: "No guarantees",
    },
  ],
  ctaPrimary: { label: "Request Demo", href: "/enterprise/demo" },
  ctaSecondary: {
    label: "Compliance Overview",
    href: "/enterprise/compliance",
  },
  ctaCopy: '"We cannot use external AI providers due to GDPR compliance requirements." — Every EU CTO and Data Protection Officer',
  gridClassName: "md:grid-cols-2 lg:grid-cols-4",
};
