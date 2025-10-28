/**
 * Community Page Props - V2 Refactored
 *
 * REUSE AUDIT (Phase 1):
 * ✅ Hero: HeroTemplate with stats row (replaced proof bullets with live stats)
 * ✅ Stats: StatsGrid molecule (replaced misused TestimonialsTemplate)
 * ✅ Contribution Types: UseCasesTemplate (kept, tightened copy)
 * ✅ How to Contribute: HowItWorks with TerminalWindow molecules (kept)
 * ✅ Support Channels: AdditionalFeaturesGrid (kept, normalized iconTone → tone)
 * ✅ Guidelines: EnterpriseCompliance (kept, improved link objects)
 * ✅ Contributors: TestimonialsTemplate (kept, renamed to "Community Voices")
 * ✅ Roadmap: EnterpriseHowItWorks (kept, tightened milestone copy)
 * ✅ FAQ: FAQTemplate with jsonLdEnabled (added)
 * ✅ Email Capture: Shared EmailCapture molecule (kept)
 *
 * SCHEMA EVOLUTION (Phase 3):
 * - Created V2 stat shapes for StatsGrid
 * - Normalized terminal blocks via HowItWorks
 * - Improved link objects in guidelines
 */

import { NetworkMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { type StatItem, StatsGrid } from '@rbee/ui/molecules/StatsGrid'
import type {
  CTATemplateProps,
  EmailCaptureProps,
  EnterpriseComplianceProps,
  EnterpriseHowItWorksProps,
  FAQTemplateProps,
  HowItWorksProps,
  TestimonialsTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import type { AdditionalFeaturesGridProps } from '@rbee/ui/templates/AdditionalFeaturesGrid/AdditionalFeaturesGrid'
import type { HeroTemplateProps } from '@rbee/ui/templates/HeroTemplate/HeroTemplateProps'
import {
  ArrowRight,
  BookOpen,
  Code2,
  FileText,
  Github,
  GitPullRequest,
  Heart,
  MessageSquare,
  Shield,
  Star,
  Users,
  Zap,
} from 'lucide-react'
import type React from 'react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

/**
 * Hero section props - Community introduction
 * Phase 4.1: Replaced proof bullets with stats row, improved media slot
 */
export const communityHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'simple',
    text: 'Open Source • GPL-3.0-or-later',
  },
  headline: {
    variant: 'simple',
    content: 'Join the rbee Community',
  },
  subcopy:
    'Connect with developers building private AI infrastructure. Contribute code, share knowledge, and help shape self-hosted AI.',
  proofElements: {
    variant: 'stats-pills',
    items: [
      { value: '500+', label: 'GitHub stars' },
      { value: '50+', label: 'Contributors' },
      { value: '1,000+', label: 'Discord members' },
    ],
    columns: 3,
  },
  ctas: {
    primary: {
      label: 'Join Discord',
      href: 'https://discord.gg/rbee',
      ariaLabel: 'Join rbee Discord community',
    },
    secondary: {
      label: 'View GitHub',
      href: 'https://github.com/veighnsche/llama-orch',
    },
  },
  helperText: 'Open source • Welcoming community • Active development',
  aside: (
    <div className="relative aspect-square w-full max-w-md">
      <NetworkMesh className="opacity-60" />
    </div>
  ),
  asideAriaLabel: 'Community network visualization showing connected developers',
}

/**
 * Hero container
 * Phase 2: Added headingId, layout, bleed, headlineLevel
 */
export const communityHeroContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'gradient-primary',
    decoration: (
      <div className="pointer-events-none absolute inset-0 opacity-25">
        <NetworkMesh />
      </div>
    ),
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
  layout: 'split',
  bleed: true,
  headingId: 'community-hero',
  headlineLevel: 1,
}

/**
 * Email capture - Join community
 * Phase 2: Added headingId
 */
export const communityEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Growing Community',
    showPulse: true,
  },
  headline: 'Stay Connected',
  subheadline: 'Get updates on new features, community events, and contribution opportunities.',
  emailInput: {
    placeholder: 'you@company.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Join Community',
  },
  trustMessage: 'No spam. Unsubscribe anytime.',
  successMessage: "Welcome! You're now part of the rbee community.",
  communityFooter: {
    text: 'Follow development on GitHub',
    linkText: 'View Repository',
    linkHref: 'https://github.com/veighnsche/llama-orch',
    subtext: 'Weekly dev notes. Roadmap issues tagged M0–M2.',
  },
  showBeeGlyphs: true,
  showIllustration: true,
}

export const communityEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
  headingId: 'newsletter',
}

/**
 * Community stats - V2 with StatsGrid
 * Phase 3 & 4.2: Replaced TestimonialsTemplate with proper StatsGrid molecule
 */
export const communityStatsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Growing Together',
  title: 'Community by the Numbers',
  description: 'Join developers building private AI infrastructure.',
  background: {
    variant: 'secondary',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'community-stats',
  divider: true,
}

/**
 * Community stats V2 - proper stat objects for StatsGrid
 */
export const communityStatsV2: StatItem[] = [
  {
    value: '500+',
    label: 'GitHub stars',
    icon: <Star className="h-5 w-5" />,
    valueTone: 'primary',
  },
  {
    value: '50+',
    label: 'Contributors',
    icon: <Users className="h-5 w-5" />,
    valueTone: 'primary',
  },
  {
    value: '200+',
    label: 'Merged PRs',
    icon: <GitPullRequest className="h-5 w-5" />,
    valueTone: 'primary',
  },
  {
    value: '1,000+',
    label: 'Discord members',
    icon: <MessageSquare className="h-5 w-5" />,
    valueTone: 'primary',
  },
]

/**
 * Render function for StatsGrid in page
 */
export function CommunityStats() {
  return <StatsGrid stats={communityStatsV2} variant="pills" columns={4} />
}

/**
 * Contribution types - Adapted from UseCasesTemplate
 * Phase 2 & 7: Added headingId, tightened description
 */
export const contributionTypesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Get Involved',
  title: 'Ways to Contribute',
  description: 'Everyone can contribute. Pick the path that fits your skills.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
  headingId: 'contribution-types',
}

/**
 * Phase 4.3: Tightened copy to one-line scenario/outcome, verb-led
 */
export const contributionTypesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <Code2 className="h-6 w-6" />,
      title: 'Code Contributions',
      scenario: 'Write Rust, TypeScript, or Vue.',
      solution: 'Pick "good first issue" and submit a PR.',
      outcome: 'Direct impact. Recognition in releases.',
    },
    {
      icon: <BookOpen className="h-6 w-6" />,
      title: 'Documentation',
      scenario: 'Improve docs and guides.',
      solution: 'Fix typos, clarify steps, add examples.',
      outcome: 'Help new users get started faster.',
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: 'Testing & QA',
      scenario: 'Test features and report bugs.',
      solution: 'Run dev builds, test edge cases.',
      outcome: 'Catch issues before release.',
    },
    {
      icon: <Heart className="h-6 w-6" />,
      title: 'Design & UX',
      scenario: 'Improve user experience.',
      solution: 'Design components, suggest improvements.',
      outcome: 'More intuitive interface.',
    },
    {
      icon: <MessageSquare className="h-6 w-6" />,
      title: 'Community Support',
      scenario: 'Help users in Discord.',
      solution: 'Answer questions, welcome new members.',
      outcome: 'Build supportive community.',
    },
    {
      icon: <Star className="h-6 w-6" />,
      title: 'Advocacy',
      scenario: 'Spread the word.',
      solution: 'Write posts, create videos, speak.',
      outcome: 'Grow the community.',
    },
  ],
  columns: 3,
}

/**
 * How to contribute - Step-by-step guide
 * Phase 2: Added headingId, divider
 */
export const howToContributeContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Getting Started',
  title: 'Your First Contribution',
  description: 'From fork to merged PR in four simple steps.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'how-to-contribute',
  divider: true,
}

export const howToContributeProps: HowItWorksProps = {
  steps: [
    {
      label: 'Fork & Clone',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>git clone https://github.com/YOUR_USERNAME/llama-orch.git</div>
            <div className="text-[var(--syntax-comment)]">cd llama-orch</div>
          </>
        ),
        copyText: 'git clone https://github.com/YOUR_USERNAME/llama-orch.git\ncd llama-orch',
      },
    },
    {
      label: 'Create Branch',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>git checkout -b feature/your-feature-name</div>
            <div className="text-[var(--syntax-comment)]"># Make your changes</div>
          </>
        ),
        copyText: 'git checkout -b feature/your-feature-name',
      },
    },
    {
      label: 'Commit & Push',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>git add .</div>
            <div>git commit -m "feat: add your feature"</div>
            <div className="text-[var(--syntax-comment)]">git push origin feature/your-feature-name</div>
          </>
        ),
        copyText: 'git add .\ngit commit -m "feat: add your feature"\ngit push origin feature/your-feature-name',
      },
    },
    {
      label: 'Open Pull Request',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div># Visit GitHub and click "New Pull Request"</div>
            <div className="text-[var(--syntax-comment)]"># Fill in the PR template</div>
            <div className="text-[var(--syntax-comment)]"># Wait for review</div>
          </>
        ),
        copyText: '# Visit GitHub and click "New Pull Request"',
      },
    },
  ],
}

/**
 * Support channels - Adapted from AdditionalFeaturesGrid
 * Phase 2 & 7: Added headingId, tightened description
 */
export const supportChannelsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Get Help',
  title: 'Support Channels',
  description: 'Ways to get help, ask questions, and connect.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'support-channels',
}

/**
 * Phase 4.5 & 7: Tightened labels to ≤6 words
 */
export const supportChannelsProps: AdditionalFeaturesGridProps = {
  rows: [
    {
      categoryLabel: 'Support',
      cards: [
        {
          icon: <Github className="h-6 w-6" />,
          iconTone: 'chart-2' as const,
          title: 'GitHub Discussions',
          subtitle: 'Ask questions and discuss features.',
          href: 'https://github.com/veighnsche/llama-orch/discussions',
          ariaLabel: 'Visit GitHub Discussions',
          borderColor: 'border-chart-2/20',
        },
        {
          icon: <MessageSquare className="h-6 w-6" />,
          iconTone: 'chart-3' as const,
          title: 'Discord Server',
          subtitle: 'Real-time chat with maintainers.',
          href: 'https://discord.gg/rbee',
          ariaLabel: 'Join Discord Server',
          borderColor: 'border-chart-3/20',
        },
        {
          icon: <BookOpen className="h-6 w-6" />,
          iconTone: 'primary' as const,
          title: 'Documentation',
          subtitle: 'Guides, API references, tutorials.',
          href: '/docs',
          ariaLabel: 'Read Documentation',
          borderColor: 'border-primary/20',
        },
        {
          icon: <FileText className="h-6 w-6" />,
          iconTone: 'muted' as const,
          title: 'GitHub Issues',
          subtitle: 'Report bugs and track progress.',
          href: 'https://github.com/veighnsche/llama-orch/issues',
          ariaLabel: 'View GitHub Issues',
          borderColor: 'border-muted/20',
        },
      ],
    },
  ],
}

/**
 * Community guidelines - Adapted from EnterpriseCompliance
 * Phase 2 & 7: Added headingId, divider, tightened description
 */
export const communityGuidelinesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Community Standards',
  title: 'Guidelines & Policies',
  description: 'Our commitment to a welcoming, inclusive, productive community.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'guidelines',
  divider: true,
}

export const communityGuidelinesProps: EnterpriseComplianceProps = {
  pillars: [
    {
      icon: <Shield className="h-8 w-8" />,
      title: 'Code of Conduct',
      subtitle: 'Our commitment to a welcoming community',
      titleId: 'code-of-conduct',
      bullets: [
        { title: 'Be respectful and inclusive' },
        { title: 'No harassment or discrimination' },
        { title: 'Constructive feedback only' },
        { title: 'Help create a welcoming environment' },
      ],
      box: {
        heading: 'Read Full Code of Conduct',
        items: ['https://github.com/veighnsche/llama-orch/blob/main/CODE_OF_CONDUCT.md'],
        checkmarkColor: 'chart-3',
        disabledCheckmarks: true,
      },
    },
    {
      icon: <GitPullRequest className="h-8 w-8" />,
      title: 'Contributing Guidelines',
      subtitle: 'How to contribute effectively',
      titleId: 'contributing-guidelines',
      bullets: [
        { title: 'Follow the PR template' },
        { title: 'Write clear commit messages' },
        { title: 'Add tests for new features' },
        { title: 'Update documentation' },
      ],
      box: {
        heading: 'Read Contributing Guide',
        items: ['https://github.com/veighnsche/llama-orch/blob/main/CONTRIBUTING.md'],
        checkmarkColor: 'chart-2',
        disabledCheckmarks: true,
      },
    },
    {
      icon: <FileText className="h-8 w-8" />,
      title: 'License',
      subtitle: 'Open source forever',
      titleId: 'license',
      bullets: [
        { title: 'GPL-3.0-or-later license' },
        { title: 'Open source forever' },
        { title: 'Commercial use allowed' },
        { title: 'Share improvements' },
      ],
      box: {
        heading: 'Read License',
        items: ['https://github.com/veighnsche/llama-orch/blob/main/LICENSE'],
        checkmarkColor: 'chart-1',
        disabledCheckmarks: true,
      },
    },
  ],
}

/**
 * Featured contributors - Adapted from TestimonialsTemplate
 * Phase 2: Added headingId, renamed to "Community Voices"
 */
export const featuredContributorsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Community Heroes',
  title: 'Community Voices',
  description: 'Meet the developers building rbee and shaping its future.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'featured-contributors',
}

export const featuredContributorsProps: TestimonialsTemplateProps = {
  testimonials: [
    {
      quote:
        "Building rbee has been an incredible journey. The community's feedback and contributions have been invaluable.",
      author: 'Core Team',
      role: 'Maintainers',
      avatar: '👥',
    },
    {
      quote: 'Contributing to rbee taught me Rust and helped me understand AI infrastructure at a deep level.',
      author: 'Community Contributors',
      role: 'Active developers',
      avatar: '💻',
    },
    {
      quote: 'The documentation improvements from the community have made rbee so much more accessible to new users.',
      author: 'Documentation Team',
      role: 'Writers & reviewers',
      avatar: '📚',
    },
  ],
}

/**
 * Roadmap - Adapted from EnterpriseHowItWorks
 * Phase 2 & 7: Added headingId, divider, tightened description
 */
export const roadmapContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: "What's Next",
  title: 'Project Roadmap',
  description: "Development milestones and what's coming next.",
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  headingId: 'roadmap',
  divider: true,
}

/**
 * Phase 4.8: Tightened milestone copy
 */
export const roadmapProps: EnterpriseHowItWorksProps = {
  deploymentSteps: [
    {
      index: 1,
      icon: <Zap className="h-6 w-6" />,
      title: 'M0: Foundation',
      intro: 'Core orchestration, multi-GPU, OpenAI-compatible API.',
      items: ['In Progress (68%)', 'Multi-GPU orchestration', 'OpenAI-compatible API', 'Basic CLI tools'],
    },
    {
      index: 2,
      icon: <Users className="h-6 w-6" />,
      title: 'M1: Collaboration',
      intro: 'Team workspaces, shared pools, RBAC.',
      items: ['Planned', 'Team workspaces', 'Shared GPU pools', 'Role-based access'],
    },
    {
      index: 3,
      icon: <Shield className="h-6 w-6" />,
      title: 'M2: Enterprise',
      intro: 'SOC2, audit trails, enterprise deploys.',
      items: ['Planned', 'SOC2 compliance', 'Audit logging', 'Enterprise deployment'],
    },
    {
      index: 4,
      icon: <Star className="h-6 w-6" />,
      title: 'M3: Marketplace',
      intro: 'GPU marketplace, provider earnings, decentralized compute.',
      items: ['Future', 'GPU marketplace', 'Provider earnings', 'Decentralized compute'],
    },
  ],
  timeline: {
    heading: 'Development Timeline',
    description: 'Our roadmap for the next 12 months',
    weeks: [
      { week: 'Q1 2025', phase: 'M0 Complete' },
      { week: 'Q2 2025', phase: 'M1 Beta' },
      { week: 'Q3 2025', phase: 'M2 Planning' },
      { week: 'Q4 2025', phase: 'M3 Research' },
    ],
  },
}

/**
 * FAQ section
 * Phase 2: Added headingId
 */
export const communityFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Common Questions',
  title: 'Community FAQ',
  description: 'Answers to common questions about contributing and community involvement.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
  headingId: 'community-faq',
}

/**
 * Phase 4.9: Enabled jsonLdEnabled for SEO
 */
export const communityFAQProps: FAQTemplateProps = {
  categories: ['General', 'Contributing', 'Technical', 'Community'],
  jsonLdEnabled: true,
  faqItems: [
    {
      value: 'q1',
      question: 'How do I get started contributing?',
      answer:
        'Start by reading our Contributing Guide and Code of Conduct. Then, look for issues tagged "good first issue" on GitHub. Join our Discord to ask questions and get help from the community.',
      category: 'Contributing',
    },
    {
      value: 'q2',
      question: 'Do I need to know Rust to contribute?',
      answer:
        'No! We welcome contributions in many areas: documentation, testing, design, TypeScript/Vue frontend work, and community support. Rust knowledge is helpful for core features, but not required.',
      category: 'Contributing',
    },
    {
      value: 'q3',
      question: 'How long does it take to get a PR reviewed?',
      answer:
        'We aim to review PRs within 2-3 business days. Complex changes may take longer. You can ping maintainers in Discord if your PR needs attention.',
      category: 'Contributing',
    },
    {
      value: 'q4',
      question: 'Can I use rbee commercially?',
      answer:
        'Yes! rbee is licensed under GPL-3.0-or-later, which allows commercial use. You can use rbee in your business, but if you modify and distribute it, you must share those modifications under the same license.',
      category: 'General',
    },
    {
      value: 'q5',
      question: 'How can I become a maintainer?',
      answer:
        'Maintainers are community members who have made consistent, high-quality contributions over time. Show initiative, help others, and demonstrate expertise in specific areas. We promote from within.',
      category: 'Community',
    },
    {
      value: 'q6',
      question: 'Where can I get help with my contribution?',
      answer:
        'Join our Discord server for real-time help, or post in GitHub Discussions for longer-form questions. The community is friendly and eager to help new contributors.',
      category: 'Community',
    },
  ],
}

/**
 * Final CTA
 * Phase 2: Container already has gradient-primary and center align (correct)
 */
export const communityCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'gradient-primary',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
}

export const communityCTAProps: CTATemplateProps = {
  eyebrow: 'Ready to Contribute?',
  title: 'Join the rbee Community Today',
  subtitle: 'Help build the future of private AI infrastructure. Every contribution matters.',
  primary: {
    label: 'Start Contributing',
    href: 'https://github.com/veighnsche/llama-orch',
    iconRight: ArrowRight,
  },
  secondary: {
    label: 'Join Discord',
    href: 'https://discord.gg/rbee',
  },
  note: 'Open source • GPL-3.0-or-later • Welcoming community',
  emphasis: 'gradient',
  align: 'center',
}
