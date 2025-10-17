import type { CTATemplateProps, FAQTemplateProps, HeroTemplateProps } from '@rbee/ui/templates'
import { FileText, Mail, Shield } from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Privacy Hero ===

/**
 * Privacy Hero - Simple legal page hero
 */
export const privacyHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    icon: <Shield className="size-6" />,
    text: 'Privacy Policy',
  },
  headline: {
    variant: 'simple',
    content: 'Your Privacy, Your Control',
  },
  subcopy:
    'Transparent privacy practices. GDPR-compliant. Your data stays yours. Last updated: October 17, 2025.',
  proofElements: {
    variant: 'assurance',
    items: [
      { text: 'GDPR Compliant', icon: <Shield className="h-4 w-4" /> },
      { text: 'EU Data Residency', icon: <FileText className="h-4 w-4" /> },
      { text: 'No Third-Party Tracking', icon: <Shield className="h-4 w-4" /> },
    ],
  },
  ctas: {
    primary: {
      label: 'Contact Privacy Team',
      href: '#contact',
    },
    secondary: {
      label: 'Download Policy (PDF)',
      href: '/legal/privacy.pdf',
    },
  },
  helperText: 'Effective Date: October 17, 2025 • Version 1.0',
  aside: null,
  background: {
    variant: 'honeycomb',
    size: 'small',
    fadeDirection: 'bottom',
  },
  padding: 'compact',
}

// === Privacy FAQ ===

/**
 * Privacy FAQ - All privacy topics as Q&A for easy navigation
 */
export const privacyFAQProps: FAQTemplateProps = {
  badgeText: 'Privacy Policy',
  categories: ['General', 'Data Collection', 'Your Rights', 'Security', 'Legal'],
  faqItems: [
    // General
    {
      value: 'scope',
      question: 'Who does this Privacy Policy apply to?',
      answer: (
        <div className="space-y-4">
          <p>
            This Privacy Policy applies to all users of rbee software and services, including developers, GPU
            providers, and enterprise customers.
          </p>
          <p>
            <strong>Scope:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>rbee open-source software (GPL-3.0-or-later)</li>
            <li>rbee.ai website and documentation</li>
            <li>rbee marketplace services (if applicable)</li>
            <li>Support and community platforms</li>
          </ul>
        </div>
      ),
      category: 'General',
    },
    {
      value: 'self-hosted',
      question: 'What about self-hosted deployments?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Self-hosted rbee instances are under YOUR control.</strong> When you self-host rbee, you are the
            data controller. This policy covers only data we collect through rbee.ai and related services.
          </p>
          <p>
            For self-hosted deployments:
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>You control all data processing</li>
            <li>You set your own privacy policies</li>
            <li>No data is sent to rbee.ai unless you explicitly configure telemetry</li>
            <li>You are responsible for GDPR compliance in your deployment</li>
          </ul>
        </div>
      ),
      category: 'General',
    },
    {
      value: 'commitment',
      question: "What is rbee's privacy commitment?",
      answer: (
        <div className="space-y-4">
          <p>
            rbee is built with privacy by design. As an open-source, self-hosted platform, we believe you should have
            complete control over your data.
          </p>
          <p>
            <strong>Our commitments:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Transparency:</strong> Open-source code means you can verify our privacy practices
            </li>
            <li>
              <strong>Data minimization:</strong> We collect only what's necessary
            </li>
            <li>
              <strong>EU data residency:</strong> Data processed in EU data centers only
            </li>
            <li>
              <strong>No selling:</strong> We never sell your data to third parties
            </li>
            <li>
              <strong>User rights:</strong> Full GDPR rights (access, deletion, portability)
            </li>
          </ul>
        </div>
      ),
      category: 'General',
    },

    // Data Collection
    {
      value: 'what-data',
      question: 'What data do you collect?',
      answer: (
        <div className="space-y-4">
          <p>We collect minimal data necessary to provide our services:</p>
          <p>
            <strong>1. Account Information (if you create an account):</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Email address</li>
            <li>Username</li>
            <li>Password (hashed with bcrypt/argon2)</li>
          </ul>
          <p>
            <strong>2. Usage Data (anonymous):</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Page views and navigation patterns</li>
            <li>Feature usage statistics</li>
            <li>Error logs and crash reports</li>
          </ul>
          <p>
            <strong>3. Technical Data:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>IP address (for security and rate limiting)</li>
            <li>Browser type and version</li>
            <li>Operating system</li>
            <li>Device type (desktop, mobile, tablet)</li>
          </ul>
          <p>
            <strong>4. Optional Telemetry (self-hosted only, opt-in):</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Deployment configuration (anonymized)</li>
            <li>Performance metrics</li>
            <li>Error reports</li>
          </ul>
        </div>
      ),
      category: 'Data Collection',
    },
    {
      value: 'how-collect',
      question: 'How do you collect data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>1. Direct Collection:</strong> Information you provide when creating an account, contacting
            support, or subscribing to updates.
          </p>
          <p>
            <strong>2. Automatic Collection:</strong> Technical data collected via server logs and analytics (Plausible
            Analytics - privacy-focused, GDPR-compliant, no cookies).
          </p>
          <p>
            <strong>3. Opt-in Telemetry:</strong> Self-hosted instances can optionally send anonymized usage data to
            help us improve rbee. This is disabled by default.
          </p>
        </div>
      ),
      category: 'Data Collection',
    },
    {
      value: 'why-collect',
      question: 'Why do you collect data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Legal Basis (GDPR Article 6):</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Legitimate Interest:</strong> Analytics and error tracking to improve the platform
            </li>
            <li>
              <strong>Consent:</strong> Optional telemetry (explicit opt-in)
            </li>
            <li>
              <strong>Contract:</strong> Account data necessary to provide services
            </li>
            <li>
              <strong>Legal Obligation:</strong> Compliance with EU regulations
            </li>
          </ul>
          <p>
            <strong>Purposes:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Provide and maintain rbee services</li>
            <li>Improve software quality and user experience</li>
            <li>Respond to support requests</li>
            <li>Detect and prevent security threats</li>
            <li>Comply with legal obligations</li>
          </ul>
        </div>
      ),
      category: 'Data Collection',
    },

    // Your Rights
    {
      value: 'gdpr-rights',
      question: 'What are my GDPR rights?',
      answer: (
        <div className="space-y-4">
          <p>Under GDPR, you have the following rights:</p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Right to Access (Art. 15):</strong> Request a copy of your personal data
            </li>
            <li>
              <strong>Right to Rectification (Art. 16):</strong> Correct inaccurate data
            </li>
            <li>
              <strong>Right to Erasure (Art. 17):</strong> Request deletion of your data ("right to be forgotten")
            </li>
            <li>
              <strong>Right to Data Portability (Art. 20):</strong> Receive your data in machine-readable format
            </li>
            <li>
              <strong>Right to Restrict Processing (Art. 18):</strong> Limit how we use your data
            </li>
            <li>
              <strong>Right to Object (Art. 21):</strong> Object to data processing based on legitimate interests
            </li>
            <li>
              <strong>Right to Withdraw Consent:</strong> Withdraw consent for optional data processing
            </li>
          </ul>
          <p>
            <strong>How to exercise your rights:</strong> Email privacy@rbee.ai with your request. We will respond
            within 30 days.
          </p>
        </div>
      ),
      category: 'Your Rights',
    },
    {
      value: 'delete-data',
      question: 'How do I delete my data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Account Deletion:</strong>
          </p>
          <ol className="list-decimal space-y-2 pl-6">
            <li>Log in to your account</li>
            <li>Go to Settings → Account → Delete Account</li>
            <li>Confirm deletion</li>
          </ol>
          <p>
            <strong>What happens when you delete your account:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Personal data is permanently deleted within 30 days</li>
            <li>Anonymized usage data may be retained for analytics</li>
            <li>Legal records retained for 7 years (GDPR compliance)</li>
          </ul>
          <p>
            <strong>Request deletion via email:</strong> If you cannot access your account, email privacy@rbee.ai with
            your account details.
          </p>
        </div>
      ),
      category: 'Your Rights',
    },
    {
      value: 'data-portability',
      question: 'How do I export my data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Self-Service Export:</strong>
          </p>
          <ol className="list-decimal space-y-2 pl-6">
            <li>Log in to your account</li>
            <li>Go to Settings → Privacy → Export Data</li>
            <li>Download your data as JSON or CSV</li>
          </ol>
          <p>
            <strong>What's included:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>Account information</li>
            <li>Configuration settings</li>
            <li>Usage history</li>
            <li>Support tickets</li>
          </ul>
          <p>
            <strong>Request via email:</strong> Email privacy@rbee.ai if you need assistance exporting your data.
          </p>
        </div>
      ),
      category: 'Your Rights',
    },

    // Security
    {
      value: 'data-security',
      question: 'How do you protect my data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Security Measures:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Encryption:</strong> TLS 1.3 for data in transit, AES-256 for data at rest
            </li>
            <li>
              <strong>Authentication:</strong> Bcrypt/Argon2 password hashing, optional 2FA
            </li>
            <li>
              <strong>Access Controls:</strong> Role-based access control (RBAC), principle of least privilege
            </li>
            <li>
              <strong>Audit Logging:</strong> Immutable audit trail for all data access
            </li>
            <li>
              <strong>Regular Audits:</strong> Security audits and penetration testing
            </li>
            <li>
              <strong>Incident Response:</strong> 24-hour breach notification (GDPR Art. 33)
            </li>
          </ul>
        </div>
      ),
      category: 'Security',
    },
    {
      value: 'data-breach',
      question: 'What happens if there\'s a data breach?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>GDPR Article 33 Compliance:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Notification:</strong> We will notify the supervisory authority within 72 hours
            </li>
            <li>
              <strong>User Notification:</strong> Affected users notified without undue delay
            </li>
            <li>
              <strong>Transparency:</strong> Public disclosure of breach details (if required)
            </li>
            <li>
              <strong>Remediation:</strong> Immediate action to contain and resolve the breach
            </li>
          </ul>
          <p>
            <strong>Incident Response Plan:</strong>
          </p>
          <ol className="list-decimal space-y-2 pl-6">
            <li>Detect and contain breach</li>
            <li>Assess impact and affected data</li>
            <li>Notify authorities and users</li>
            <li>Remediate vulnerabilities</li>
            <li>Post-incident review and improvements</li>
          </ol>
        </div>
      ),
      category: 'Security',
    },

    // Legal
    {
      value: 'data-retention',
      question: 'How long do you keep my data?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Retention Periods:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Account Data:</strong> Until account deletion + 30 days
            </li>
            <li>
              <strong>Usage Data:</strong> 13 months (analytics)
            </li>
            <li>
              <strong>Audit Logs:</strong> 7 years (GDPR compliance)
            </li>
            <li>
              <strong>Support Tickets:</strong> 3 years
            </li>
            <li>
              <strong>Legal Records:</strong> 7 years (legal obligation)
            </li>
          </ul>
          <p>
            <strong>Automated Deletion:</strong> Data is automatically deleted after retention periods expire.
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'third-parties',
      question: 'Do you share data with third parties?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>We do NOT sell your data.</strong>
          </p>
          <p>
            <strong>Limited Third-Party Services:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Plausible Analytics:</strong> Privacy-focused analytics (GDPR-compliant, no cookies, EU-hosted)
            </li>
            <li>
              <strong>Email Service:</strong> Transactional emails (account verification, password reset)
            </li>
            <li>
              <strong>Payment Processor:</strong> Stripe (if using paid services, PCI-DSS compliant)
            </li>
          </ul>
          <p>
            <strong>Data Processing Agreements:</strong> All third parties sign GDPR-compliant Data Processing
            Agreements (DPAs).
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'international-transfers',
      question: 'Do you transfer data outside the EU?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>EU Data Residency:</strong> All data is processed and stored in EU data centers (Frankfurt,
            Amsterdam, Paris).
          </p>
          <p>
            <strong>No US Cloud Dependencies:</strong> We do not use US cloud providers (AWS US, GCP US, Azure US).
          </p>
          <p>
            <strong>Standard Contractual Clauses:</strong> If international transfers are necessary (e.g., support
            requests), we use EU Standard Contractual Clauses (SCCs) per GDPR Article 46.
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'cookies',
      question: 'Do you use cookies?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Minimal Cookie Usage:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Essential Cookies:</strong> Session management, authentication (necessary for service)
            </li>
            <li>
              <strong>No Tracking Cookies:</strong> We do not use third-party tracking cookies
            </li>
            <li>
              <strong>No Advertising Cookies:</strong> We do not use advertising cookies
            </li>
          </ul>
          <p>
            <strong>Analytics:</strong> We use Plausible Analytics, which does not use cookies.
          </p>
          <p>
            <strong>Cookie Management:</strong> You can disable cookies in your browser settings. Note that disabling
            essential cookies may affect functionality.
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'children',
      question: 'Do you collect data from children?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Age Restriction:</strong> rbee services are not intended for children under 16 years old.
          </p>
          <p>
            <strong>No Knowingly Collection:</strong> We do not knowingly collect personal data from children under 16.
          </p>
          <p>
            <strong>Parental Notice:</strong> If you believe we have collected data from a child under 16, please
            contact privacy@rbee.ai immediately. We will delete the data within 48 hours.
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'policy-changes',
      question: 'How will I know if this policy changes?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Notification Methods:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Email:</strong> Notification sent to registered email address
            </li>
            <li>
              <strong>Website Banner:</strong> Prominent notice on rbee.ai
            </li>
            <li>
              <strong>Version History:</strong> All changes documented with dates
            </li>
          </ul>
          <p>
            <strong>Material Changes:</strong> For significant changes, we will provide 30 days' notice and require
            re-acceptance.
          </p>
          <p>
            <strong>Version History:</strong> Previous versions available at /legal/privacy/history
          </p>
        </div>
      ),
      category: 'Legal',
    },
    {
      value: 'supervisory-authority',
      question: 'How do I file a complaint with a supervisory authority?',
      answer: (
        <div className="space-y-4">
          <p>
            <strong>Right to Lodge Complaint (GDPR Article 77):</strong>
          </p>
          <p>
            If you believe we have violated your privacy rights, you have the right to lodge a complaint with your
            local Data Protection Authority.
          </p>
          <p>
            <strong>EU Data Protection Authorities:</strong>
          </p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Netherlands:</strong> Autoriteit Persoonsgegevens (AP) - autoriteitpersoonsgegevens.nl
            </li>
            <li>
              <strong>Germany:</strong> Bundesbeauftragte für den Datenschutz und die Informationsfreiheit (BfDI) -
              bfdi.bund.de
            </li>
            <li>
              <strong>France:</strong> Commission Nationale de l'Informatique et des Libertés (CNIL) - cnil.fr
            </li>
            <li>
              <strong>Full List:</strong> edpb.europa.eu/about-edpb/about-edpb/members_en
            </li>
          </ul>
          <p>
            <strong>Contact Us First:</strong> We encourage you to contact us first at privacy@rbee.ai so we can
            address your concerns.
          </p>
        </div>
      ),
      category: 'Legal',
    },
  ],
  searchPlaceholder: 'Search privacy topics…',
  emptySearchKeywords: ['GDPR', 'data', 'rights', 'security', 'cookies'],
  expandAllLabel: 'Expand all',
  collapseAllLabel: 'Collapse all',
  supportCard: {
    title: 'Privacy Questions?',
    links: [
      { label: 'Email Privacy Team', href: 'mailto:privacy@rbee.ai' },
      { label: 'View Compliance Docs', href: '/docs/compliance' },
      { label: 'Data Processing Agreement', href: '/legal/dpa' },
    ],
    cta: {
      label: 'Contact Support',
      href: '/contact',
    },
  },
  jsonLdEnabled: true,
}

// === Privacy CTA ===

/**
 * Privacy CTA - Contact privacy team
 */
export const privacyCTAProps: CTATemplateProps = {
  eyebrow: 'Questions?',
  title: 'Need Help with Privacy?',
  subtitle: 'Our privacy team is here to help. Contact us for data requests, questions, or concerns.',
  primary: {
    label: 'Contact Privacy Team',
    href: 'mailto:privacy@rbee.ai',
    iconLeft: Mail,
  },
  secondary: {
    label: 'View Compliance Docs',
    href: '/docs/compliance',
    iconLeft: FileText,
  },
  note: 'We respond to all privacy requests within 30 days (GDPR requirement)',
  align: 'center',
  emphasis: 'none',
}
