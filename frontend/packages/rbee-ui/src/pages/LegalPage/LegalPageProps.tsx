import { DeploymentFlow, EuLedgerGrid, SectorGrid, SecurityMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  ComparisonTemplateProps,
  EmailCaptureProps,
  EnterpriseCTAProps,
  EnterpriseHeroProps,
  EnterpriseHowItWorksProps,
  EnterpriseSecurityProps,
  FAQTemplateProps,
  ProblemTemplateProps,
  ProvidersEarningsGPUModel,
  ProvidersEarningsPreset,
  ProvidersEarningsProps,
  SolutionTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  BookOpen,
  Building2,
  Calendar,
  CheckCircle,
  Clock,
  Eye,
  FileCheck,
  FileSearch,
  FileText,
  FileX,
  Gavel,
  Globe,
  Heart,
  KeyRound,
  Lock,
  MessageSquare,
  Rocket,
  Scale,
  Search,
  Server,
  Shield,
  Users,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Legal Industry Hero ===

export const legalHeroProps: EnterpriseHeroProps = {
  badge: {
    icon: <Scale className="size-6" />,
    text: 'Legal Tech',
  },
  heading: 'AI-Powered Legal Research and Document Review for Law Firms',
  description:
    'Transform legal workflows with private AI infrastructure. Process contracts, case law, and discovery documents at scale—without sending client data to third-party APIs.',
  stats: [
    {
      value: '95%',
      label: 'Faster Document Review',
      helpText: 'Accelerate contract analysis and due diligence with AI-powered document understanding',
    },
    {
      value: '100%',
      label: 'Client Confidentiality',
      helpText: 'All processing on-premises or in your private cloud—zero data leaves your control',
    },
    {
      value: '24/7',
      label: 'Research Assistant',
      helpText: 'Query case law, statutes, and internal precedents instantly with semantic search',
    },
  ],
  primaryCta: {
    text: 'Schedule Legal Tech Demo',
    ariaLabel: 'Schedule a demo for legal industry solutions',
  },
  secondaryCta: {
    text: 'Download Case Study',
    href: '#case-study',
  },
  helperText: 'Trusted by law firms for confidential client work. No data sharing, no API calls to external LLMs.',
  complianceChips: [
    {
      icon: <Lock className="h-3 w-3" />,
      label: 'Attorney-Client Privilege',
      ariaLabel: 'Maintains attorney-client privilege with on-premises processing',
    },
    {
      icon: <Shield className="h-3 w-3" />,
      label: 'Bar Association Compliant',
      ariaLabel: 'Compliant with legal ethics and bar association rules',
    },
    {
      icon: <FileCheck className="h-3 w-3" />,
      label: 'Audit Trail',
      ariaLabel: 'Complete audit trail for all AI-assisted work product',
    },
  ],
  auditConsole: {
    title: 'Legal Research Query',
    badge: 'Private',
    filterButtons: [
      { label: 'All', ariaLabel: 'Filter: All events', active: true },
      { label: 'Query', ariaLabel: 'Filter: Query events' },
      { label: 'Research', ariaLabel: 'Filter: Research events' },
      { label: 'Audit', ariaLabel: 'Filter: Audit events' },
    ],
    events: [
      {
        event: 'legal.research_query',
        user: 'partner@lawfirm.com',
        time: '2025-10-17T14:23:41Z',
        displayTime: '2025-10-17 14:23:41 UTC',
        status: 'success',
      },
      {
        event: 'legal.precedent_retrieval',
        user: 'system',
        time: '2025-10-17T14:23:42Z',
        displayTime: '2025-10-17 14:23:42 UTC',
        status: 'success',
      },
      {
        event: 'legal.summary_generation',
        user: 'system',
        time: '2025-10-17T14:23:45Z',
        displayTime: '2025-10-17 14:23:45 UTC',
        status: 'success',
      },
      {
        event: 'legal.matter_audit_log',
        user: 'system',
        time: '2025-10-17T14:23:46Z',
        displayTime: '2025-10-17 14:23:46 UTC',
        status: 'success',
      },
    ],
    footer: {
      retention: 'Matter-level logging',
      tamperProof: 'Attorney work product',
    },
  },
  floatingBadges: [
    {
      label: 'Queries',
      value: '1.2M processed',
      ariaLabel: '1.2 million legal queries processed',
      position: 'top-right',
    },
    {
      label: 'Confidentiality',
      value: '100% Private',
      ariaLabel: '100% private on-premises processing',
      position: 'bottom-left',
    },
  ],
}

// === Email Capture ===

export const legalEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Legal Tech',
    showPulse: true,
  },
  headline: 'Get the Legal AI Playbook',
  subheadline:
    'Learn how leading law firms use private AI for contract review, legal research, and due diligence while maintaining attorney-client privilege.',
  emailInput: {
    placeholder: 'partner@lawfirm.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Download Playbook',
  },
  trustMessage: 'Trusted by law firms for confidential client work.',
  successMessage: 'Check your inbox! Legal AI Playbook sent.',
  communityFooter: {
    text: 'Need help with legal AI?',
    linkText: 'Talk to Legal Tech Team',
    linkHref: '/contact/legal',
    subtext: 'We respond within 1 business day',
  },
}

export const legalEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}

// === Problem Template ===

export const legalProblemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Legal AI Confidentiality Crisis',
  description:
    'Law firms face an impossible choice: use public AI tools and risk ethics violations, or fall behind competitors who embrace AI.',
  kicker: 'Attorney-Client Privilege at Risk',
  kickerVariant: 'destructive',
  background: {
    variant: 'gradient-destructive',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
  ctaBanner: {
    copy: 'Deploy private AI that preserves attorney-client privilege. No data leakage, no ethics violations.',
    primary: { label: 'Schedule Legal Tech Demo', href: '/industries/legal/demo' },
    secondary: { label: 'Download Case Study', href: '/industries/legal/case-study' },
  },
}

export const legalProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: <FileX className="h-6 w-6" />,
      title: 'Confidentiality Breaches',
      body: 'Using ChatGPT or Claude for client work violates attorney-client privilege. Every query sends data to third-party servers. Once data leaves your network, you lose control. Bar associations issue warnings. Malpractice insurers take notice.',
      tone: 'destructive',
      tag: 'ABA Model Rule 1.6',
    },
    {
      icon: <Scale className="h-6 w-6" />,
      title: 'Ethics Violations',
      body: 'Bar associations require "reasonable measures" to protect client confidentiality. Public AI tools fail this standard. No encryption in transit. No access controls. No audit trails. Your license is at risk.',
      tone: 'destructive',
      tag: 'Ethics violation',
    },
    {
      icon: <Clock className="h-6 w-6" />,
      title: 'Competitive Pressure',
      body: "Associates need AI to stay competitive. Manual research takes hours. Contract review drowns teams. But firms can't risk client data exposure. The result? Shadow IT, compliance gaps, and frustrated attorneys.",
      tone: 'destructive',
      tag: 'Productivity crisis',
    },
  ],
}

// === Solution Template ===

export const legalSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
}

export const legalSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Lock className="size-6" />,
      title: 'Zero Data Leakage',
      body:
        'Models run on your servers. No API calls, no cloud uploads, no third-party access. Client data stays in your four walls.',
    },
    {
      icon: <FileSearch className="size-6" />,
      title: 'Firm Knowledge Base',
      body:
        'Index your precedents, memos, and case files. AI searches your internal documents first—results are tailored to your practice.',
    },
    {
      icon: <Eye className="size-6" />,
      title: 'Audit Trail for Work Product',
      body:
        'Every AI query is logged with matter number, attorney, and timestamp. Demonstrate reasonable care for malpractice defense.',
    },
    {
      icon: <Users className="size-6" />,
      title: 'Multi-User Access Control',
      body:
        "Role-based permissions ensure associates only access matters they're assigned to. Partners control model access.",
    },
  ],
}

// === Legal Use Cases ===

export const legalUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'How Law Firms Use Private AI',
  description: 'Real-world applications across litigation, corporate, and transactional practices.',
  eyebrow: 'Legal Workflows',
}

export const legalUseCasesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <FileSearch className="size-8" />,
      title: 'Contract Review & Due Diligence',
      scenario:
        'M&A teams need to review hundreds of contracts in days. Manual review is slow, and public AI tools risk confidentiality.',
      solution:
        'Upload contracts to your private AI. Extract key terms (indemnification, termination, change of control) in minutes. Flag unusual clauses for partner review.',
      outcome: '10x faster contract review. Zero data sent to external APIs. Full audit trail for client files.',
    },
    {
      icon: <Search className="size-8" />,
      title: 'Legal Research & Precedent Search',
      scenario:
        "Associates spend hours searching case law and firm precedents. Public AI hallucinates citations and can't access your internal memos.",
      solution:
        "Index your firm's knowledge base (briefs, memos, case files). Ask natural language questions: \"Find all cases where we successfully argued force majeure in NY.\"",
      outcome: 'Instant access to firm precedents. AI cites actual documents, not hallucinated cases. Research time cut by 70%.',
    },
    {
      icon: <FileText className="size-8" />,
      title: 'Discovery Document Analysis',
      scenario:
        'Litigation teams drown in discovery documents. Keyword search misses context, and manual review is prohibitively expensive.',
      solution:
        'Process discovery documents with semantic search. Ask: "Find all emails discussing the 2022 acquisition." AI understands synonyms and context.',
      outcome: 'Reduce discovery review costs by 80%. Find smoking-gun documents faster. Maintain privilege over work product.',
    },
  ],
  columns: 3,
}

// === Legal How It Works ===

export const legalHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Deploy Private Legal AI in 3 Steps',
  description: 'From pilot to firm-wide rollout in weeks, not months.',
  eyebrow: 'Implementation',
}

export const legalHowItWorksProps: EnterpriseHowItWorksProps = {
  deploymentSteps: [
    {
      index: 1,
      icon: <Server className="size-6" />,
      title: 'Install on Firm Servers',
      intro:
        'Deploy rbee on your existing hardware or private cloud. No data leaves your network. IT team maintains full control.',
      items: [
        'Works with on-premises servers or AWS/Azure private VPC',
        'Integrates with Active Directory for single sign-on',
        'Supports GPU acceleration for faster document processing',
      ],
    },
    {
      index: 2,
      icon: <FileSearch className="size-6" />,
      title: 'Index Firm Knowledge Base',
      intro:
        'Upload precedents, memos, and case files. AI creates semantic index for instant retrieval. Respects matter-level permissions.',
      items: [
        'Supports PDF, DOCX, and scanned documents (OCR)',
        'Automatic metadata extraction (matter number, author, date)',
        'Role-based access control per matter',
      ],
    },
    {
      index: 3,
      icon: <Users className="size-6" />,
      title: 'Train Associates & Partners',
      intro:
        'One-hour training session covers prompt engineering for legal research, contract review workflows, and audit trail requirements.',
      items: [
        'Best practices for AI-assisted legal research',
        'How to verify AI-generated citations',
        'Documenting AI use for client billing',
      ],
    },
  ],
  timeline: {
    heading: 'Typical Deployment Timeline',
    description: 'From pilot to firm-wide rollout',
    weeks: [
      { week: 'Week 1-2', phase: 'Hardware setup and knowledge base indexing' },
      { week: 'Week 3', phase: 'Pilot with single practice group' },
      { week: 'Week 4-8', phase: 'Firm-wide rollout and training' },
    ],
  },
}

// === Legal Security ===

export const legalSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Built for Attorney-Client Privilege and Bar Association Rules',
  description:
    'Every security control designed to meet legal ethics requirements. Demonstrate reasonable care for client confidentiality.',
  eyebrow: 'Security & Ethics',
}

export const legalSecurityProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Lock className="size-8" />,
      title: 'On-Premises Processing',
      subtitle: 'ABA Model Rule 1.6 Compliant',
      intro:
        'Models run on your hardware. Client data never transmitted to external APIs. Satisfies ABA Model Rule 1.6 (confidentiality).',
      bullets: [
        'Zero network calls to third-party LLM providers',
        'All embeddings and inference happen locally',
        'Air-gapped deployment option for maximum security',
      ],
      docsHref: '/docs/security/on-premises',
    },
    {
      icon: <FileCheck className="size-8" />,
      title: 'Audit Trail for Work Product',
      subtitle: 'Malpractice Defense Ready',
      intro:
        'Every AI query logged with matter number, attorney, timestamp, and model used. Demonstrate competence for malpractice defense.',
      bullets: [
        'Tamper-evident logs with cryptographic hashing',
        'Export audit reports per matter for client files',
        '7-year retention policy (matches legal malpractice statute of limitations)',
      ],
      docsHref: '/docs/security/audit-trail',
    },
    {
      icon: <Users className="size-8" />,
      title: 'Matter-Level Access Control',
      subtitle: 'DMS Integration',
      intro:
        "Associates only access AI for matters they're assigned to. Partners approve model usage per practice group.",
      bullets: [
        'Integrates with document management systems (iManage, NetDocuments)',
        'Role-based permissions (partner, associate, paralegal)',
        'Conflict check integration to prevent inadvertent access',
      ],
      docsHref: '/docs/security/access-control',
    },
    {
      icon: <Eye className="size-8" />,
      title: 'Citation Verification',
      subtitle: 'Prevent Hallucinations',
      intro:
        'AI provides source documents for every claim. Associates verify citations before including in briefs or memos.',
      bullets: [
        'Direct links to source documents in firm knowledge base',
        'Confidence scores for AI-generated summaries',
        'Flagging system for hallucinated or uncertain claims',
      ],
      docsHref: '/docs/security/citations',
    },
  ],
}

// === Legal Comparison ===

export const legalComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Public AI vs. Private Legal AI',
  description: 'Why law firms choose on-premises AI over ChatGPT and Claude.',
  eyebrow: 'Comparison',
}

export const legalComparisonProps: ComparisonTemplateProps = {
  columns: [
    { key: 'ChatGPT/Claude', label: 'ChatGPT/Claude' },
    { key: 'rbee', label: 'rbee', accent: true },
  ],
  rows: [
    {
      feature: 'Data stays on-premises',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Attorney-client privilege protected',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Meets ABA Model Rule 1.6',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Audit trail for work product',
      values: {
        'ChatGPT/Claude': 'partial',
        rbee: 'yes',
      },
    },
    {
      feature: 'Firm knowledge base search',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Citation verification',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Matter-level permissions',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
    {
      feature: 'Conflict check integration',
      values: {
        'ChatGPT/Claude': 'no',
        rbee: 'yes',
      },
    },
  ],
}

// === Legal ROI Calculator ===

export const legalROICalculatorContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Calculate Your Legal AI ROI',
  description:
    'See how much time and money your firm saves by replacing manual research and contract review with private AI.',
  eyebrow: 'ROI Calculator',
}

const legalGPUModels: ProvidersEarningsGPUModel[] = [
  { name: 'Llama-3.1-8B (Contract Review)', baseRate: 0.50, vram: 16 },
  { name: 'Llama-3.1-70B (Legal Research)', baseRate: 2.00, vram: 80 },
  { name: 'Llama-3.1-405B (Complex Analysis)', baseRate: 8.00, vram: 400 },
]

const legalPresets: ProvidersEarningsPreset[] = [
  {
    label: 'Small Firm (10 attorneys)',
    hours: 8,
    utilization: 40,
  },
  {
    label: 'Mid-Size Firm (50 attorneys)',
    hours: 12,
    utilization: 60,
  },
  {
    label: 'Large Firm (200+ attorneys)',
    hours: 24,
    utilization: 80,
  },
]

export const legalROICalculatorProps: ProvidersEarningsProps = {
  gpuModels: legalGPUModels,
  presets: legalPresets,
  commission: 0.15,
  configTitle: 'Configuration',
  selectGPULabel: 'Select Model',
  presetsLabel: 'Quick Presets',
  hoursLabel: 'Hours per Day',
  utilizationLabel: 'Utilization %',
  earningsTitle: 'Time Savings',
  monthlyLabel: 'Monthly',
  basedOnText: (hours: number, utilization: number) =>
    `Based on ${hours}h/day at ${utilization}% utilization`,
  takeHomeLabel: 'Hours Saved',
  dailyLabel: 'Daily',
  yearlyLabel: 'Yearly',
  breakdownTitle: 'Breakdown',
  hourlyRateLabel: 'Hourly Rate',
  hoursPerMonthLabel: 'Hours/Month',
  utilizationBreakdownLabel: 'Utilization',
  commissionLabel: 'Platform Fee',
  yourTakeHomeLabel: 'Your Savings',
  ctaLabel: 'Schedule ROI Review',
  ctaAriaLabel: 'Schedule a call to review ROI calculations',
  secondaryCTALabel: 'Download Report',
  formatCurrency: (n: number) => `$${n.toLocaleString()}`,
  formatHourly: (n: number) => `${n.toFixed(2)}h`,
}

// === Legal FAQ ===

export const legalFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Legal AI Questions Answered',
  description: 'Common questions from law firms about private AI deployment.',
  eyebrow: 'FAQ',
}

export const legalFAQProps: FAQTemplateProps = {
  badgeText: 'FAQ',
  categories: ['All', 'Ethics', 'Technical', 'Billing'],
  faqItems: [
    {
      value: 'q1',
      question: 'Does using AI violate attorney-client privilege?',
      answer:
        'Not if the AI runs on your own infrastructure. Public AI tools (ChatGPT, Claude) send data to third-party servers, which can waive privilege. rbee runs entirely on your hardware—client data never leaves your network, preserving privilege.',
      category: 'Ethics',
    },
    {
      value: 'q2',
      question: 'What do bar associations say about AI in legal practice?',
      answer:
        'Most bar associations (ABA, state bars) require "reasonable measures" to protect client confidentiality (Model Rule 1.6). Using public AI tools for client work generally fails this standard. On-premises AI like rbee meets the requirement because data stays under your control.',
      category: 'Ethics',
    },
    {
      value: 'q3',
      question: 'Can AI hallucinate case citations in legal research?',
      answer:
        "Yes—public LLMs often invent fake case citations. rbee mitigates this by searching your firm's knowledge base first and providing direct links to source documents. Associates still verify citations before including them in briefs, as required by ethical rules.",
      category: 'Technical',
    },
    {
      value: 'q4',
      question: 'How do we bill clients for AI-assisted work?',
      answer:
        "Most firms bill AI-assisted work at normal hourly rates, since the attorney still reviews and approves all output. Some firms disclose AI use in engagement letters. rbee's audit trail helps document attorney supervision for billing disputes.",
      category: 'Billing',
    },
    {
      value: 'q5',
      question: 'What hardware do we need to run legal AI?',
      answer:
        'For small firms (10-20 attorneys), a single GPU server (NVIDIA RTX 4090 or A6000) is sufficient. Mid-size firms (50+ attorneys) typically deploy 2-4 GPU servers. Large firms may use a dedicated AI cluster. We help right-size hardware during the pilot phase.',
      category: 'Technical',
    },
    {
      value: 'q6',
      question: 'Can we integrate AI with our document management system?',
      answer:
        'Yes—rbee integrates with iManage, NetDocuments, and other DMS platforms. This enables matter-level access control (associates only access AI for their assigned matters) and automatic metadata extraction (matter number, client, date).',
      category: 'Technical',
    },
    {
      value: 'q7',
      question: 'How long does deployment take?',
      answer:
        'Pilot deployments take 2-4 weeks (hardware setup, knowledge base indexing, training). Firm-wide rollout adds another 4-8 weeks depending on size. Most firms start with a single practice group (e.g., M&A team) before expanding.',
      category: 'Technical',
    },
    {
      value: 'q8',
      question: 'What about malpractice insurance and AI use?',
      answer:
        'Malpractice insurers increasingly ask about AI policies. Using public AI tools without safeguards can increase premiums or void coverage. On-premises AI with audit trails demonstrates reasonable care and may reduce risk. Check with your carrier.',
      category: 'Ethics',
    },
  ],
}

// === Legal CTA ===

export const legalCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Deploy Private Legal AI Without Violating Attorney-Client Privilege',
  description:
    'Join law firms using rbee for confidential contract review, legal research, and discovery analysis. Schedule a demo to see how private AI preserves client confidentiality while accelerating legal workflows.',
  eyebrow: 'Get Started',
}

export const legalCTAProps: EnterpriseCTAProps = {
  trustStats: [
    { value: '100%', label: 'Client Confidentiality' },
    { value: '95%', label: 'Faster Document Review' },
    { value: '24/7', label: 'Research Assistant' },
    { value: 'ABA', label: 'Compliant' },
  ],
  ctaOptions: [
    {
      icon: <Calendar className="size-6" />,
      title: 'Schedule Legal Tech Demo',
      body: 'See how private AI preserves attorney-client privilege while accelerating legal workflows.',
      buttonText: 'Book Demo',
      buttonHref: '/demo/legal',
      buttonVariant: 'default',
      buttonAriaLabel: 'Schedule a demo for legal industry solutions',
      tone: 'primary',
    },
    {
      icon: <FileText className="size-6" />,
      title: 'Download Case Study',
      body: 'Learn how law firms use rbee for confidential contract review and legal research.',
      buttonText: 'Get Case Study',
      buttonHref: '#case-study',
      buttonVariant: 'outline',
      tone: 'outline',
    },
    {
      icon: <MessageSquare className="size-6" />,
      title: 'Talk to Legal Tech Team',
      body: 'Discuss your firm\'s specific requirements with our legal AI specialists.',
      buttonText: 'Contact Us',
      buttonHref: '/contact/legal',
      buttonVariant: 'outline',
      tone: 'outline',
    },
  ],
}
