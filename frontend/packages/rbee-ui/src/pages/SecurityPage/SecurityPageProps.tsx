import { SecurityMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  CTATemplateProps,
  EmailCaptureProps,
  EnterpriseComplianceProps,
  EnterpriseHeroProps,
  EnterpriseHowItWorksProps,
  EnterpriseSecurityProps,
  ErrorHandlingTemplateProps,
  FAQTemplateProps,
  HowItWorksProps,
  ProblemTemplateProps,
  SecurityIsolationProps,
  SolutionTemplateProps,
  TechnicalTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  Bug,
  CheckCircle,
  Clock,
  Eye,
  FileCheck,
  FileText,
  GitBranch,
  KeyRound,
  Lock,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Users,
  XCircle,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Security Hero ===

/**
 * Security Hero - Main hero section with security console visual
 */
export const securityHeroProps: EnterpriseHeroProps = {
  badge: {
    icon: <Shield className="size-6" />,
    text: 'Security by Design',
  },
  heading: 'Defense-in-Depth Security Architecture',
  description:
    'Zero-trust principles. Process isolation. Immutable audit trails. Built with security at every layer—from input validation to cryptographic guarantees.',
  stats: [
    {
      value: '6 Crates',
      label: 'Security Modules',
      helpText: 'Dedicated security crates for auth, audit, validation, secrets, JWT, and deadlines',
    },
    {
      value: '32 Types',
      label: 'Audit Events',
      helpText: 'Comprehensive audit event types tracked across all operations',
    },
    {
      value: 'Zero Trust',
      label: 'Architecture',
      helpText: 'Never trust, always verify—security at every boundary',
    },
  ],
  primaryCta: {
    text: 'View Security Docs',
    ariaLabel: 'View security documentation',
  },
  secondaryCta: {
    text: 'Report Vulnerability',
    href: '#vulnerability-disclosure',
  },
  helperText: 'Open-source security. Auditable by design. Community-reviewed.',
  complianceChips: [
    {
      icon: <Shield className="h-3 w-3" />,
      label: 'Zero Trust',
      ariaLabel: 'Zero Trust Architecture',
    },
    {
      icon: <Lock className="h-3 w-3" />,
      label: 'Process Isolation',
      ariaLabel: 'Process Isolation Security',
    },
    {
      icon: <FileCheck className="h-3 w-3" />,
      label: 'Immutable Logs',
      ariaLabel: 'Immutable Audit Logs',
    },
  ],
  auditConsole: {
    title: 'Security Event Monitor',
    badge: 'Live',
    filterButtons: [
      { label: 'All', ariaLabel: 'Filter: All events', active: true },
      { label: 'Auth', ariaLabel: 'Filter: Auth events' },
      { label: 'Threats', ariaLabel: 'Filter: Threat events' },
      { label: 'Audit', ariaLabel: 'Filter: Audit events' },
    ],
    events: [
      {
        event: 'threat.detected',
        user: 'security-scanner',
        time: '2025-10-17T19:45:23Z',
        displayTime: '2025-10-17 19:45:23 UTC',
        status: 'warning',
      },
      {
        event: 'auth.mfa_success',
        user: 'admin@company.eu',
        time: '2025-10-17T19:44:15Z',
        displayTime: '2025-10-17 19:44:15 UTC',
        status: 'success',
      },
      {
        event: 'input.validation_failed',
        user: 'api-client-7',
        time: '2025-10-17T19:43:08Z',
        displayTime: '2025-10-17 19:43:08 UTC',
        status: 'error',
      },
      {
        event: 'secrets.rotation',
        user: 'secrets-manager',
        time: '2025-10-17T19:42:00Z',
        displayTime: '2025-10-17 19:42:00 UTC',
        status: 'success',
      },
    ],
    footer: {
      retention: 'Retention: 7 years',
      tamperProof: 'Hash-chained',
    },
  },
  floatingBadges: [
    {
      label: 'Threat Detection',
      value: 'Active',
      ariaLabel: 'Real-time threat detection active',
      position: 'top-right',
    },
    {
      label: 'Security Crates',
      value: '6 Modules',
      ariaLabel: '6 dedicated security crates',
      position: 'bottom-left',
    },
  ],
}

// === Email Capture ===

/**
 * Email Capture - Security whitepaper download
 */
export const securityEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Security Whitepaper',
    showPulse: false,
  },
  headline: 'Get the Security Architecture Whitepaper',
  subheadline:
    'Deep dive into our defense-in-depth architecture, threat model, and security guarantees. 24 pages of technical detail.',
  emailInput: {
    placeholder: 'security-engineer@company.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Download Whitepaper',
  },
  trustMessage: 'Open-source. Auditable. Community-reviewed.',
  successMessage: 'Check your email! The whitepaper is on its way.',
  communityFooter: {
    text: 'Prefer to read online?',
    linkText: 'View Security Docs',
    linkHref: '/docs/security',
    subtext: 'Full documentation available on GitHub',
  },
}

/**
 * Email capture container - Background wrapper
 */
export const securityEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === Threat Model (Problem Template) ===

/**
 * Threat Model - Security threats we defend against
 */
export const securityThreatModelProps: ProblemTemplateProps = {
  items: [
    {
      icon: <ShieldAlert className="size-6" />,
      title: 'Prompt Injection',
      body: 'Malicious prompts attempting to bypass security controls or extract sensitive data.',
      tone: 'destructive',
      tag: 'High Risk',
    },
    {
      icon: <Zap className="size-6" />,
      title: 'Resource Exhaustion',
      body: 'DoS attacks attempting to exhaust GPU memory, CPU, or network resources.',
      tone: 'destructive',
      tag: 'High Risk',
    },
    {
      icon: <Eye className="size-6" />,
      title: 'Data Leakage',
      body: 'Unauthorized access to training data, prompts, or model outputs.',
      tone: 'destructive',
      tag: 'Critical',
    },
    {
      icon: <Bug className="size-6" />,
      title: 'Model Poisoning',
      body: 'Attempts to corrupt model weights or inject backdoors during training.',
      tone: 'destructive',
      tag: 'High Risk',
    },
    {
      icon: <AlertTriangle className="size-6" />,
      title: 'Side-Channel Attacks',
      body: 'Timing attacks, cache attacks, or other side-channel information leakage.',
      tone: 'destructive',
      tag: 'Medium Risk',
    },
    {
      icon: <KeyRound className="size-6" />,
      title: 'Credential Theft',
      body: 'Attempts to steal API keys, tokens, or other authentication credentials.',
      tone: 'destructive',
      tag: 'Critical',
    },
  ],
}

/**
 * Threat model container
 */
export const securityThreatModelContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Defense Layers (Solution Template) ===

/**
 * Defense Layers - Defense-in-depth approach
 */
export const securityDefenseLayersProps: SolutionTemplateProps = {
  features: [
    {
      icon: <FileText className="size-6" />,
      title: 'Input Validation',
      body: 'Strict validation of all inputs at API boundaries. Schema enforcement. Type safety.',
    },
    {
      icon: <Shield className="size-6" />,
      title: 'Authentication & Authorization',
      body: 'Multi-factor auth. Role-based access control. JWT with short expiry. Token rotation.',
    },
    {
      icon: <Lock className="size-6" />,
      title: 'Process Isolation',
      body: 'Each worker runs in isolated process. Memory isolation. No shared state.',
    },
    {
      icon: <FileCheck className="size-6" />,
      title: 'Audit Logging',
      body: 'Immutable audit trails. Hash-chained logs. 7-year retention. Tamper-evident.',
    },
    {
      icon: <KeyRound className="size-6" />,
      title: 'Secrets Management',
      body: 'Encrypted at rest. Never logged. Auto-rotation. Zeroization on shutdown.',
    },
    {
      icon: <Clock className="size-6" />,
      title: 'Deadline Propagation',
      body: 'Request timeouts enforced. Prevents resource exhaustion. Graceful degradation.',
    },
  ],
}

/**
 * Defense layers container
 */
export const securityDefenseLayersContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Defense-in-Depth Security Architecture',
  description: 'Multiple layers of security controls protect your AI infrastructure from threats at every level.',
  kicker: 'Security Layers',
  background: {
    variant: 'background',
    decoration: (
      <div className="absolute inset-0 opacity-15">
        <SecurityMesh className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Security Crates (Enterprise Security) ===

/**
 * Security Crates - 6 dedicated security modules
 */
export const securityCratesProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Users className="size-6" />,
      title: 'auth-min',
      subtitle: 'Minimal authentication primitives',
      intro: 'Core authentication building blocks for secure user management.',
      bullets: [
        'Multi-factor authentication',
        'Role-based access control (RBAC)',
        'Session management with secure cookies',
        'Password hashing with Argon2',
        'API key generation and validation',
      ],
      docsHref: '/docs/security/auth-min',
    },
    {
      icon: <FileCheck className="size-6" />,
      title: 'audit-logging',
      subtitle: 'Immutable audit trail system',
      intro: 'Tamper-evident audit logs with blockchain-style hash chaining.',
      bullets: [
        'Hash-chained audit logs (blockchain-style)',
        '32 distinct audit event types',
        '7-year retention policy',
        'Tamper-evident verification',
        'Compliance export formats (JSON, CSV)',
      ],
      docsHref: '/docs/security/audit-logging',
    },
    {
      icon: <FileText className="size-6" />,
      title: 'input-validation',
      subtitle: 'Strict input validation and sanitization',
      intro: 'Comprehensive input validation to prevent injection attacks.',
      bullets: [
        'Schema-based validation (JSON Schema)',
        'Type safety enforcement',
        'SQL injection prevention',
        'XSS protection',
        'Path traversal prevention',
      ],
      docsHref: '/docs/security/input-validation',
    },
    {
      icon: <KeyRound className="size-6" />,
      title: 'secrets-management',
      subtitle: 'Secure secrets handling',
      intro: 'Encrypted secrets storage with automatic rotation support.',
      bullets: [
        'Encrypted at rest (AES-256-GCM)',
        'Never logged or exposed in errors',
        'Auto-rotation support',
        'Zeroization on process shutdown',
        'Environment variable isolation',
      ],
      docsHref: '/docs/security/secrets-management',
    },
    {
      icon: <ShieldCheck className="size-6" />,
      title: 'jwt-guardian',
      subtitle: 'JWT token management',
      intro: 'Secure JWT token generation, validation, and rotation.',
      bullets: [
        'Short-lived tokens (15 min default)',
        'Refresh token rotation',
        'Token revocation support',
        'Claims validation',
        'HMAC-SHA256 signing',
      ],
      docsHref: '/docs/security/jwt-guardian',
    },
    {
      icon: <Clock className="size-6" />,
      title: 'deadline-propagation',
      subtitle: 'Request timeout enforcement',
      intro: 'Distributed deadline propagation to prevent resource exhaustion.',
      bullets: [
        'Per-request deadlines',
        'Timeout propagation across services',
        'Graceful degradation',
        'Resource exhaustion prevention',
        'Cancellation support',
      ],
      docsHref: '/docs/security/deadline-propagation',
    },
  ],
}

/**
 * Security crates container
 */
export const securityCratesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Process Isolation (Security Isolation) ===

/**
 * Process Isolation - Isolation features
 */
export const securityIsolationProps: SecurityIsolationProps = {
  cratesTitle: 'Security Crates',
  cratesSubtitle: 'Six dedicated security modules for defense-in-depth',
  securityCrates: [
    { name: 'auth-min', description: 'Authentication primitives', hoverColor: 'hsl(var(--chart-1))' },
    { name: 'audit-logging', description: 'Immutable audit trails', hoverColor: 'hsl(var(--chart-2))' },
    { name: 'input-validation', description: 'Input sanitization', hoverColor: 'hsl(var(--chart-3))' },
    { name: 'secrets-management', description: 'Encrypted secrets', hoverColor: 'hsl(var(--chart-4))' },
    { name: 'jwt-guardian', description: 'JWT token management', hoverColor: 'hsl(var(--chart-5))' },
    { name: 'deadline-propagation', description: 'Timeout enforcement', hoverColor: 'hsl(var(--chart-1))' },
  ],
  processIsolationTitle: 'Process Isolation',
  processIsolationSubtitle: 'Each worker runs in its own isolated process',
  processFeatures: [
    { title: 'Process-level isolation with separate memory space', color: 'chart-3' },
    { title: 'No shared memory between workers', color: 'chart-3' },
    { title: 'Sandboxed execution with minimal privileges', color: 'chart-3' },
    { title: 'Fail-safe defaults with automatic restart', color: 'chart-3' },
  ],
  zeroTrustTitle: 'Zero Trust Architecture',
  zeroTrustSubtitle: 'Never trust, always verify at every boundary',
  zeroTrustFeatures: [
    { title: 'All network traffic authenticated and encrypted', color: 'chart-2' },
    { title: 'No implicit trust between components', color: 'chart-2' },
    { title: 'Continuous verification of identity and context', color: 'chart-2' },
    { title: 'Least privilege access enforcement', color: 'chart-2' },
  ],
}

/**
 * Process isolation container
 */
export const securityIsolationContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Security Guarantees (Enterprise Compliance) ===

/**
 * Security Guarantees - Cryptographic and timing guarantees
 */
export const securityGuaranteesProps: EnterpriseComplianceProps = {
  pillars: [
    {
      icon: <Lock className="size-8" />,
      title: 'Timing-Safe Operations',
      subtitle: 'Constant-time operations to prevent timing attacks',
      titleId: 'timing-safe',
      bullets: [
        { title: 'Constant-time comparisons for secrets' },
        { title: 'No timing side-channels in auth' },
        { title: 'Constant-time password verification' },
        { title: 'Timing-safe token validation' },
      ],
      box: {
        heading: 'Cryptographic Primitives',
        items: [
          'AES-256-GCM for encryption',
          'Argon2id for password hashing',
          'HMAC-SHA256 for signatures',
          'ChaCha20-Poly1305 for streams',
        ],
        checkmarkColor: 'chart-3',
      },
    },
    {
      icon: <KeyRound className="size-8" />,
      title: 'Zeroization',
      subtitle: 'Secure memory cleanup to prevent data leakage',
      titleId: 'zeroization',
      bullets: [
        { title: 'Secrets zeroed on process shutdown' },
        { title: 'Memory wiped after use' },
        { title: 'No secrets in core dumps' },
        { title: 'Secure deallocation' },
      ],
      box: {
        heading: 'Security Testing',
        items: ['Fuzzing with cargo-fuzz', 'Property-based testing', 'Security unit tests', 'Penetration testing'],
        checkmarkColor: 'chart-2',
      },
    },
    {
      icon: <Eye className="size-8" />,
      title: 'Anti-Fingerprinting',
      subtitle: 'Minimal information disclosure to prevent reconnaissance',
      titleId: 'anti-fingerprinting',
      bullets: [
        { title: 'No version leakage in errors' },
        { title: 'Generic error messages' },
        { title: 'No stack traces to clients' },
        { title: 'Minimal information disclosure' },
      ],
      box: {
        heading: 'Security Monitoring',
        items: [
          'Dependabot vulnerability scanning',
          'cargo-audit dependency checks',
          'Security advisory tracking',
          'Incident response procedures',
        ],
        checkmarkColor: 'chart-1',
      },
    },
  ],
}

/**
 * Security guarantees container
 */
export const securityGuaranteesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Security Development Lifecycle (Enterprise How It Works) ===

/**
 * Security Development Lifecycle - How we build secure software
 */
export const securitySDLCProps: EnterpriseHowItWorksProps = {
  deploymentSteps: [
    {
      index: 1,
      icon: <FileText className="size-6" />,
      title: 'Threat Modeling',
      intro: 'Identify threats early. Document attack vectors. Design defenses before coding.',
      items: ['STRIDE methodology', 'Attack trees', 'Security requirements'],
    },
    {
      index: 2,
      icon: <GitBranch className="size-6" />,
      title: 'Secure Coding',
      intro: 'Follow secure coding guidelines. Use safe APIs. Avoid common vulnerabilities.',
      items: ['Rust memory safety', 'Type safety', 'Linting with Clippy'],
    },
    {
      index: 3,
      icon: <CheckCircle className="size-6" />,
      title: 'Security Testing',
      intro: 'Comprehensive security testing. Fuzzing. Static analysis. Penetration testing.',
      items: ['cargo-fuzz', 'cargo-audit', 'SAST tools', 'Manual review'],
    },
    {
      index: 4,
      icon: <Shield className="size-6" />,
      title: 'Continuous Monitoring',
      intro: 'Monitor for vulnerabilities. Patch quickly. Audit logs reviewed regularly.',
      items: ['Dependabot', 'Security advisories', 'Incident response plan'],
    },
  ],
  timeline: {
    heading: 'Security Lifecycle',
    description: 'Continuous security throughout development',
    weeks: [
      { week: 'Week 1', phase: 'Threat Modeling' },
      { week: 'Week 2-3', phase: 'Secure Development' },
      { week: 'Week 4', phase: 'Security Testing' },
      { week: 'Ongoing', phase: 'Monitoring' },
    ],
  },
}

/**
 * Security SDLC container
 */
export const securitySDLCContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Vulnerability Disclosure (How It Works) ===

/**
 * Vulnerability Disclosure - How to report security issues
 */
export const securityVulnerabilityDisclosureProps: HowItWorksProps = {
  steps: [
    {
      label: 'Report via GitHub Security Advisories',
      number: 1,
      block: {
        kind: 'code',
        language: 'bash',
        code: '# Navigate to the repository\n# Click "Security" tab → "Advisories" → "New draft security advisory"\n# Fill out the form with vulnerability details',
      },
    },
    {
      label: 'Include Detailed Information',
      number: 2,
      block: {
        kind: 'code',
        language: 'markdown',
        code: '## Vulnerability Report\n\n**Affected Component:** [crate/module name]\n**Severity:** [Critical/High/Medium/Low]\n**Description:** [detailed description]\n**Steps to Reproduce:** [numbered steps]\n**Impact:** [what can an attacker do?]\n**Suggested Fix:** [optional]',
      },
    },
    {
      label: 'We Respond Within 48 Hours',
      number: 3,
      block: {
        kind: 'note',
        content:
          'Our security team will acknowledge your report within 48 hours and provide an initial assessment within 7 days.',
      },
    },
    {
      label: 'Coordinated Disclosure',
      number: 4,
      block: {
        kind: 'note',
        content:
          'We will work with you to understand the issue, develop a fix, and coordinate public disclosure. You will be credited in the security advisory.',
      },
    },
  ],
}

/**
 * Vulnerability disclosure container
 */
export const securityVulnerabilityDisclosureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === Security Architecture (Technical Template) ===

/**
 * Security Architecture - Technical deep-dive
 */
export const securityArchitectureProps: TechnicalTemplateProps = {
  architectureHighlights: [
    {
      title: 'Zero-Trust Network',
      details: ['All network traffic authenticated and encrypted', 'No implicit trust between components'],
    },
    {
      title: 'Defense-in-Depth',
      details: ['Multiple layers of security', 'If one layer fails, others remain'],
    },
    {
      title: 'Immutable Audit Logs',
      details: ['All security events logged immutably', 'Tamper-evident verification'],
    },
  ],
  techStack: [
    { name: 'auth-min', description: 'Minimal auth primitives', ariaLabel: 'Authentication crate' },
    { name: 'jwt-guardian', description: 'JWT token management', ariaLabel: 'JWT crate' },
    { name: 'audit-logging', description: 'Immutable audit trails', ariaLabel: 'Audit logging crate' },
    { name: 'input-validation', description: 'Input sanitization', ariaLabel: 'Input validation crate' },
    { name: 'secrets-management', description: 'Secure secrets handling', ariaLabel: 'Secrets management crate' },
    { name: 'deadline-propagation', description: 'Timeout enforcement', ariaLabel: 'Deadline propagation crate' },
  ],
  stackLinks: {
    githubUrl: 'https://github.com/rbee/rbee',
    license: 'GPL-3.0-or-later',
    architectureUrl: '/docs/security/architecture',
  },
}

/**
 * Security architecture container
 */
export const securityArchitectureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Security FAQs ===

/**
 * Security FAQs - Common security questions
 */
export const securityFAQsProps: FAQTemplateProps = {
  badgeText: 'Security FAQs',
  categories: ['General', 'Technical', 'Compliance', 'Reporting'],
  faqItems: [
    {
      value: 'production-ready',
      question: 'Is rbee secure for production use?',
      answer:
        'Yes. rbee is designed with security as a first-class concern. We use defense-in-depth, process isolation, immutable audit logs, and cryptographic guarantees. However, security is a shared responsibility—you must also follow security best practices in your deployment.',
      category: 'General',
    },
    {
      value: 'secrets-handling',
      question: 'How do you handle secrets and API keys?',
      answer:
        'Secrets are encrypted at rest using AES-256-GCM, never logged, and zeroized on process shutdown. We support auto-rotation and environment variable isolation. See our secrets-management crate documentation for details.',
      category: 'Technical',
    },
    {
      value: 'worker-compromise',
      question: 'What happens if a worker is compromised?',
      answer:
        'Each worker runs in an isolated process with no shared memory. If one worker is compromised, it cannot access other workers or the orchestrator. The compromised worker can be terminated and restarted with a clean state.',
      category: 'Technical',
    },
    {
      value: 'mfa-support',
      question: 'Do you support multi-factor authentication?',
      answer:
        'Yes. Our auth-min crate supports multi-factor authentication (MFA) with TOTP (Time-based One-Time Password). You can require MFA for all users or specific roles.',
      category: 'Technical',
    },
    {
      value: 'audit-retention',
      question: 'How long are audit logs retained?',
      answer:
        'Audit logs are retained for 7 years by default, meeting GDPR requirements. Logs are immutable and hash-chained for tamper-evidence. You can export logs in JSON or CSV format for compliance reporting.',
      category: 'Compliance',
    },
    {
      value: 'encryption-algorithms',
      question: 'What encryption algorithms do you use?',
      answer:
        'We use industry-standard algorithms: AES-256-GCM for encryption, Argon2id for password hashing, HMAC-SHA256 for signatures, and ChaCha20-Poly1305 for stream encryption. All implementations are from audited Rust crates.',
      category: 'Technical',
    },
    {
      value: 'prompt-injection',
      question: 'How do you prevent prompt injection attacks?',
      answer:
        'We use strict input validation at API boundaries, schema enforcement, and type safety. Prompts are sanitized and validated before being sent to workers. We also support custom validation rules and content filtering.',
      category: 'Technical',
    },
    {
      value: 'code-audit',
      question: 'Is the codebase audited?',
      answer:
        'rbee is open-source and community-reviewed. We use automated security scanning (cargo-audit, Dependabot) and encourage security researchers to review our code. We plan to conduct formal security audits before 1.0 release.',
      category: 'Reporting',
    },
  ],
}

/**
 * Security FAQs container
 */
export const securityFAQsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === Security Error Handling ===

/**
 * Security Error Handling - Threat detection and incident response
 */
export const securityErrorHandlingProps: ErrorHandlingTemplateProps = {
  statusKPIs: [
    {
      icon: <ShieldCheck className="size-6" />,
      color: 'chart-3',
      label: 'Threats Blocked',
      value: '100%',
    },
    {
      icon: <Clock className="size-6" />,
      color: 'primary',
      label: 'Detection Time',
      value: '<100ms',
    },
    {
      icon: <FileCheck className="size-6" />,
      color: 'chart-2',
      label: 'Audit Events',
      value: '32 Types',
    },
  ],
  terminalContent: (
    <>
      <div className="text-muted-foreground">
        <span className="text-destructive">14:23:41</span> [SECURITY] Invalid JWT signature detected
      </div>
      <div className="text-muted-foreground">
        <span className="text-destructive">14:23:41</span> [AUDIT] Request rejected: authentication_failed
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-2">14:23:41</span> [AUDIT] Event logged: auth_rejection
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-3">14:23:41</span> [SECURITY] Connection terminated
      </div>
      <div className="text-muted-foreground">
        <span className="text-primary">14:23:42</span> [AUDIT] Immutable log written to audit trail
      </div>
    </>
  ),
  terminalFooter: (
    <span className="text-xs text-muted-foreground">
      Zero-trust validation. Fail-secure by default. All security events logged to immutable audit trail.
    </span>
  ),
  playbookCategories: [
    {
      icon: <ShieldAlert className="size-5" />,
      color: 'warning',
      title: 'Authentication Failures',
      checkCount: 6,
      severityDots: ['destructive', 'destructive', 'primary', 'primary', 'chart-2', 'chart-3'],
      description: 'Invalid tokens, expired credentials, signature mismatches',
      checks: [
        {
          severity: 'destructive',
          title: 'Invalid JWT Signature',
          meaning: 'Token signature verification failed',
          actionLabel: 'Reject & Log',
          href: '/docs/security/jwt-validation',
          guideLabel: 'Security Guide',
          guideHref: '/docs/security/authentication',
        },
        {
          severity: 'destructive',
          title: 'Expired Token',
          meaning: 'Token past expiration deadline',
          actionLabel: 'Reject & Audit',
          href: '/docs/security/token-expiry',
          guideLabel: 'Token Management',
          guideHref: '/docs/security/tokens',
        },
        {
          severity: 'primary',
          title: 'Missing Authorization',
          meaning: 'No auth header provided',
          actionLabel: 'Return 401',
          href: '/docs/security/missing-auth',
          guideLabel: 'Auth Setup',
          guideHref: '/docs/security/setup',
        },
      ],
    },
    {
      icon: <XCircle className="size-5" />,
      color: 'primary',
      title: 'Input Validation',
      checkCount: 5,
      severityDots: ['destructive', 'primary', 'primary', 'chart-2', 'chart-3'],
      description: 'Malformed requests, injection attempts, invalid schemas',
      checks: [
        {
          severity: 'destructive',
          title: 'Schema Validation Failed',
          meaning: 'Request body does not match schema',
          actionLabel: 'Reject & Log',
          href: '/docs/security/schema-validation',
          guideLabel: 'Validation Guide',
          guideHref: '/docs/security/input-validation',
        },
        {
          severity: 'primary',
          title: 'Path Traversal Attempt',
          meaning: 'Suspicious path characters detected',
          actionLabel: 'Block & Audit',
          href: '/docs/security/path-traversal',
          guideLabel: 'Security Hardening',
          guideHref: '/docs/security/hardening',
        },
      ],
    },
    {
      icon: <Lock className="size-5" />,
      color: 'chart-2',
      title: 'Access Control',
      checkCount: 4,
      severityDots: ['destructive', 'primary', 'chart-2', 'chart-3'],
      description: 'Unauthorized access, privilege escalation, resource limits',
      checks: [
        {
          severity: 'destructive',
          title: 'Unauthorized Resource Access',
          meaning: 'User lacks permission for resource',
          actionLabel: 'Deny & Log',
          href: '/docs/security/authorization',
          guideLabel: 'Access Control',
          guideHref: '/docs/security/rbac',
        },
        {
          severity: 'primary',
          title: 'Rate Limit Exceeded',
          meaning: 'Request quota exhausted',
          actionLabel: 'Throttle',
          href: '/docs/security/rate-limiting',
          guideLabel: 'Rate Limits',
          guideHref: '/docs/security/quotas',
        },
      ],
    },
    {
      icon: <FileCheck className="size-5" />,
      color: 'chart-3',
      title: 'Audit Trail',
      checkCount: 3,
      severityDots: ['primary', 'chart-2', 'chart-3'],
      description: 'Log integrity, retention, compliance events',
      checks: [
        {
          severity: 'primary',
          title: 'Audit Log Write Failed',
          meaning: 'Cannot write to immutable log',
          actionLabel: 'Fail Request',
          href: '/docs/security/audit-logging',
          guideLabel: 'Audit Setup',
          guideHref: '/docs/security/audit-trail',
        },
      ],
    },
  ],
}

/**
 * Security Error Handling container
 */
export const securityErrorHandlingContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Threat Detection & Incident Response',
  description: 'Fail-secure error handling with comprehensive audit trails',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Final CTA ===

/**
 * Final CTA - Review security docs
 */
export const securityCTAProps: CTATemplateProps = {
  eyebrow: 'Security Documentation',
  title: 'Review Our Security Architecture',
  subtitle: 'Dive deep into our security crates, threat model, and security guarantees.',
  primary: {
    label: 'View Security Docs',
    href: '/docs/security',
  },
  secondary: {
    label: 'Report Vulnerability',
    href: 'https://github.com/rbee/rbee/security/advisories/new',
  },
  note: 'Open-source. Auditable. Community-reviewed.',
  emphasis: 'gradient',
  align: 'center',
}

/**
 * Final CTA container
 */
export const securityCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
}
