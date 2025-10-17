import { NetworkMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
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

// Props Objects (in visual order matching page composition)
// ============================================================================

/**
 * Hero section props - Community introduction
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
    'Connect with developers building private AI infrastructure. Contribute code, share knowledge, and help shape the future of self-hosted AI.',
  proofElements: {
    variant: 'bullets',
    items: [{ title: '100% open source' }, { title: 'Welcoming community' }, { title: 'Active development' }],
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
  helperText: 'Star on GitHub • Active Contributors • Weekly Updates',
  aside: <div className="relative aspect-square w-full max-w-md">{/* Community visualization placeholder */}</div>,
  asideAriaLabel: 'Community network visualization',
}

/**
 * Hero container
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
}

/**
 * Email capture - Join community
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
}

/**
 * Community stats - Adapted from TestimonialsTemplate
 */
export const communityStatsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Growing Together',
  title: 'Community by the Numbers',
  description: 'Join developers worldwide building the future of private AI infrastructure.',
  background: {
    variant: 'secondary',
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
}

export const communityStatsProps: TestimonialsTemplateProps = {
  testimonials: [
    {
      quote: '500+',
      author: 'GitHub Stars',
      role: 'Growing daily',
      avatar: '⭐',
    },
    {
      quote: '50+',
      author: 'Contributors',
      role: 'From 12 countries',
      avatar: '👥',
    },
    {
      quote: '200+',
      author: 'Pull Requests',
      role: 'Merged this year',
      avatar: '🔀',
    },
    {
      quote: '1,000+',
      author: 'Discord Members',
      role: 'Active discussions',
      avatar: '💬',
    },
  ],
}

/**
 * Contribution types - Adapted from UseCasesTemplate
 */
export const contributionTypesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Get Involved',
  title: 'Ways to Contribute',
  description: 'Everyone can contribute to rbee. Find the path that fits your skills and interests.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

export const contributionTypesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <Code2 className="h-6 w-6" />,
      title: 'Code Contributions',
      scenario: 'Write Rust, TypeScript, or Vue code to improve rbee.',
      solution: 'Pick an issue tagged "good first issue" or "help wanted" and submit a PR.',
      outcome: 'Direct impact on the project. Recognition in release notes.',
    },
    {
      icon: <BookOpen className="h-6 w-6" />,
      title: 'Documentation',
      scenario: 'Help others understand rbee by improving docs.',
      solution: 'Fix typos, clarify instructions, add examples, or write guides.',
      outcome: 'Make rbee more accessible. Help new users get started faster.',
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: 'Testing & QA',
      scenario: 'Test new features, report bugs, and verify fixes.',
      solution: 'Run dev builds, test edge cases, and provide detailed bug reports.',
      outcome: 'Improve stability. Catch issues before release.',
    },
    {
      icon: <Heart className="h-6 w-6" />,
      title: 'Design & UX',
      scenario: 'Improve the user experience and visual design.',
      solution: 'Design UI components, create mockups, or suggest UX improvements.',
      outcome: 'Better user experience. More intuitive interface.',
    },
    {
      icon: <MessageSquare className="h-6 w-6" />,
      title: 'Community Support',
      scenario: 'Help other users in Discord and GitHub Discussions.',
      solution: 'Answer questions, share tips, and welcome new members.',
      outcome: 'Build a supportive community. Help others succeed.',
    },
    {
      icon: <Star className="h-6 w-6" />,
      title: 'Advocacy',
      scenario: 'Spread the word about rbee.',
      solution: 'Write blog posts, create videos, or speak at meetups.',
      outcome: 'Grow the community. Reach more developers.',
    },
  ],
  columns: 3,
}

/**
 * How to contribute - Step-by-step guide
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
 */
export const supportChannelsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Get Help',
  title: 'Support Channels',
  description: 'Multiple ways to get help, ask questions, and connect with the community.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const supportChannelsProps: AdditionalFeaturesGridProps = {
  rows: [
    {
      categoryLabel: 'Support',
      cards: [
        {
          icon: <Github className="h-6 w-6" />,
          iconTone: 'chart-2' as const,
          title: 'GitHub Discussions',
          subtitle: 'Ask questions, share ideas, and discuss features with the community.',
          href: 'https://github.com/veighnsche/llama-orch/discussions',
          ariaLabel: 'Visit GitHub Discussions',
          borderColor: 'border-chart-2/20',
        },
        {
          icon: <MessageSquare className="h-6 w-6" />,
          iconTone: 'chart-3' as const,
          title: 'Discord Server',
          subtitle: 'Real-time chat with maintainers and community members.',
          href: 'https://discord.gg/rbee',
          ariaLabel: 'Join Discord Server',
          borderColor: 'border-chart-3/20',
        },
        {
          icon: <BookOpen className="h-6 w-6" />,
          iconTone: 'primary' as const,
          title: 'Documentation',
          subtitle: 'Comprehensive guides, API references, and tutorials.',
          href: '/docs',
          ariaLabel: 'Read Documentation',
          borderColor: 'border-primary/20',
        },
        {
          icon: <FileText className="h-6 w-6" />,
          iconTone: 'muted' as const,
          title: 'GitHub Issues',
          subtitle: 'Report bugs, request features, and track development progress.',
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
 */
export const communityGuidelinesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Community Standards',
  title: 'Guidelines & Policies',
  description: 'Our commitment to a welcoming, inclusive, and productive community.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
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
 */
export const featuredContributorsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: 'Community Heroes',
  title: 'Featured Contributors',
  description: 'Meet the developers building rbee and shaping its future.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
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
 */
export const roadmapContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: "What's Next",
  title: 'Project Roadmap',
  description: "Our development milestones and what's coming next.",
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const roadmapProps: EnterpriseHowItWorksProps = {
  deploymentSteps: [
    {
      index: 1,
      icon: <Zap className="h-6 w-6" />,
      title: 'M0: Foundation',
      intro: 'Core orchestration, multi-GPU support, and OpenAI-compatible API.',
      items: ['In Progress (68%)', 'Multi-GPU orchestration', 'OpenAI-compatible API', 'Basic CLI tools'],
    },
    {
      index: 2,
      icon: <Users className="h-6 w-6" />,
      title: 'M1: Collaboration',
      intro: 'Team features, shared pools, and role-based access control.',
      items: ['Planned', 'Team workspaces', 'Shared GPU pools', 'Role-based access'],
    },
    {
      index: 3,
      icon: <Shield className="h-6 w-6" />,
      title: 'M2: Enterprise',
      intro: 'SOC2 compliance, audit trails, and enterprise deployment options.',
      items: ['Planned', 'SOC2 compliance', 'Audit logging', 'Enterprise deployment'],
    },
    {
      index: 4,
      icon: <Star className="h-6 w-6" />,
      title: 'M3: Marketplace',
      intro: 'GPU marketplace, provider earnings, and decentralized compute.',
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
}

export const communityFAQProps: FAQTemplateProps = {
  categories: ['General', 'Contributing', 'Technical', 'Community'],
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
