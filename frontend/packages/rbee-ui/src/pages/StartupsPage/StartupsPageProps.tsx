'use client'

import { Badge } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  ComparisonTemplateProps,
  CTATemplateProps,
  EmailCaptureProps,
  FAQTemplateProps,
  ProblemTemplateProps,
  ProvidersEarningsProps,
  SolutionTemplateProps,
  TestimonialsTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import type { EnterpriseHowItWorksProps } from '@rbee/ui/templates/EnterpriseHowItWorks'
import type { HeroTemplateProps } from '@rbee/ui/templates/HeroTemplate'
import type { TechnicalTemplateProps } from '@rbee/ui/templates/TechnicalTemplate'
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Check,
  Code,
  DollarSign,
  Gauge,
  Lock,
  Rocket,
  Server,
  Shield,
  TrendingDown,
  Users,
  X,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects
// ============================================================================

// === Hero Template ===
export const startupsHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    text: '💡 Build AI Products Without Burning Cash',
    icon: <Rocket className="h-3.5 w-3.5" />,
  },
  headline: {
    variant: 'two-line-highlight',
    prefix: 'Own Your AI Stack.',
    highlight: 'Escape API Fees.',
  },
  subcopy:
    'Stop paying per token. Build your AI products on your own infrastructure from day one. Scale independently, control costs, and never worry about rate limits or vendor lock-in.',
  subcopyMaxWidth: 'wide',
  proofElements: {
    variant: 'stats-tiles',
    items: [
      {
        icon: <DollarSign className="size-6" />,
        value: '90%',
        label: 'Lower costs vs APIs',
      },
      {
        icon: <Zap className="size-6" />,
        value: '100%',
        label: 'Control & ownership',
      },
      {
        icon: <Gauge className="size-6" />,
        value: 'No',
        label: 'Rate limits',
      },
    ],
    columns: 3,
  },
  ctas: {
    primary: {
      label: 'Calculate Your Savings',
      href: '#roi-calculator',
      showIcon: true,
      dataUmamiEvent: 'cta:startups-calculate-savings',
    },
    secondary: {
      label: 'View Pricing',
      href: '/pricing',
      variant: 'outline',
    },
  },
  trustElements: {
    variant: 'text',
    text: 'OpenAI-compatible API • All GPUs supported • 100% open source',
  },
  aside: (
    <div className="relative">
      <div className="rounded border border-border bg-card p-6 shadow-lg">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            <span className="font-semibold">Cost Comparison</span>
          </div>
          <Badge variant="secondary">Monthly</Badge>
        </div>
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">OpenAI API</span>
              <span className="font-mono font-bold text-destructive">$2,400/mo</span>
            </div>
            <div className="h-2 w-full rounded-full bg-destructive/20">
              <div className="h-2 w-full rounded-full bg-destructive" />
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">rbee (self-hosted)</span>
              <span className="font-mono font-bold text-emerald-500">$240/mo</span>
            </div>
            <div className="h-2 w-full rounded-full bg-emerald-500/20">
              <div className="h-2 w-[10%] rounded-full bg-emerald-500" />
            </div>
          </div>
          <div className="rounded-md bg-emerald-500/10 p-3 text-center">
            <div className="text-2xl font-bold text-emerald-500">$2,160 saved</div>
            <div className="text-xs text-muted-foreground">90% cost reduction</div>
          </div>
        </div>
      </div>
    </div>
  ),
  asideAriaLabel: 'Cost comparison visualization showing 90% savings',
  background: {
    variant: 'gradient',
  },
}

// === Email Capture ===
export const startupsEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'secondary',
  },
  paddingY: 'lg',
  maxWidth: '3xl',
  align: 'center',
}

export const startupsEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Join 500+ Startups',
    showPulse: true,
  },
  headline: 'Start Building for Free',
  subheadline: 'Get early access to rbee and join the waitlist for our startup program.',
  emailInput: {
    placeholder: 'your@startup.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Join Waitlist',
  },
  trustMessage: 'We respect your privacy. Unsubscribe anytime.',
  successMessage: "You're on the list! Check your email for next steps.",
}

// === Problem Template ===
export const startupsProblemContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'The API Cost Trap',
  title: 'Your AI Bills Are Spiraling Out of Control',
  description: 'API providers charge per token. As you scale, costs explode—and you have zero control.',
  background: {
    variant: 'gradient-destructive',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctaBanner: {
    copy: 'Break free from the API cost spiral. Own your infrastructure and control your destiny.',
    primary: { label: 'Calculate Your Savings', href: '#roi-calculator' },
    secondary: { label: 'See How It Works', href: '#how-it-works' },
  },
}

export const startupsProblemProps: ProblemTemplateProps = {
  items: [
    {
      icon: <TrendingDown className="h-6 w-6" />,
      title: 'Unpredictable Costs',
      body: 'Started at $200/mo. Now $2,400/mo. Next month? Who knows. Your runway is burning faster than expected.',
      tag: 'Cost spiral',
      tone: 'destructive',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'Rate Limits Kill Growth',
      body: 'Hit your limit at peak usage. Customers see errors. You lose revenue. Upgrading costs even more.',
      tag: 'Growth blocker',
      tone: 'destructive',
    },
    {
      icon: <Lock className="h-6 w-6" />,
      title: 'Vendor Lock-In',
      body: 'Built on OpenAI? Switching costs are astronomical. They own your pricing, your models, your future.',
      tag: 'Zero leverage',
      tone: 'destructive',
    },
  ],
}

// === Solution Template ===
export const startupsSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'The rbee Advantage',
  title: 'Own Your Stack. Control Your Costs. Scale on Your Terms.',
  description:
    'Self-hosted AI infrastructure with OpenAI-compatible APIs. Build once, run anywhere, pay nothing per token.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctas: {
    primary: {
      label: 'Get Started Free',
      href: '/getting-started',
      ariaLabel: 'Get started with rbee',
    },
    secondary: {
      label: 'View Documentation',
      href: '/docs',
    },
  },
}

export const startupsSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <DollarSign className="h-8 w-8" aria-hidden="true" />,
      title: 'Predictable Costs',
      body: 'Pay for hardware once. No per-token fees. No surprises.',
    },
    {
      icon: <Gauge className="h-8 w-8" aria-hidden="true" />,
      title: 'Unlimited Scale',
      body: 'No rate limits. No throttling. Scale as fast as you need.',
    },
    {
      icon: <Shield className="h-8 w-8" aria-hidden="true" />,
      title: 'Full Ownership',
      body: 'Your models, your data, your infrastructure. Forever.',
    },
  ],
  steps: [
    {
      title: 'Install rbee',
      body: 'One command. Works on any hardware. CUDA, Metal, or CPU.',
    },
    {
      title: 'Drop in OpenAI SDK',
      body: 'Change one line: point to localhost. Your code works unchanged.',
    },
    {
      title: 'Deploy & Scale',
      body: 'Add more GPUs as you grow. No API bills. No limits.',
    },
    {
      title: 'Build Your Product',
      body: 'Focus on features, not infrastructure costs.',
    },
  ],
  earnings: {
    rows: [
      {
        model: 'Prototype (1 GPU)',
        meta: 'RTX 4090 • Local',
        value: '$0/mo',
        note: 'vs $200-500 API costs',
      },
      {
        model: 'MVP (2-3 GPUs)',
        meta: 'Multi-GPU • Self-hosted',
        value: '$0/mo',
        note: 'vs $1,000-2,000 API costs',
      },
      {
        model: 'Scale (5+ GPUs)',
        meta: 'Distributed cluster',
        value: '$0/mo',
        note: 'vs $5,000+ API costs',
      },
    ],
    disclaimer: 'Hardware costs are one-time. API costs are recurring forever.',
  },
}

// === ROI Calculator (adapted ProvidersEarnings) ===
export const startupsROICalculatorContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Do The Math',
  title: 'Calculate Your Savings',
  description: "See how much you'll save by switching from API providers to self-hosted rbee.",
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsROICalculatorProps: ProvidersEarningsProps = {
  gpuModels: [
    { name: 'OpenAI API (gpt-4)', baseRate: 0.03, vram: 0 }, // $0.03 per 1K tokens
    { name: 'Anthropic Claude', baseRate: 0.025, vram: 0 },
    { name: 'Both Providers', baseRate: 0.055, vram: 0 },
  ],
  presets: [
    { label: 'Light', hours: 100, utilization: 50 }, // 100K tokens/mo
    { label: 'Medium', hours: 500, utilization: 70 }, // 500K tokens/mo
    { label: 'Heavy', hours: 2000, utilization: 90 }, // 2M tokens/mo
  ],
  commission: 0.1, // Self-hosted cost is ~10% of API cost
  configTitle: 'Your Current API Usage',
  selectGPULabel: 'Current Provider',
  presetsLabel: 'Usage Level',
  hoursLabel: 'API Requests per Month (in thousands)',
  utilizationLabel: 'Growth Factor (%)',
  earningsTitle: 'Your Savings',
  monthlyLabel: 'Monthly Savings',
  basedOnText: (hours, utilization) => `Based on ${hours}K requests/mo at ${utilization}% growth`,
  takeHomeLabel: 'Net Savings (after hardware)',
  dailyLabel: 'Daily Savings',
  yearlyLabel: 'Yearly Savings',
  breakdownTitle: 'Cost Breakdown',
  hourlyRateLabel: 'API Cost per 1K tokens',
  hoursPerMonthLabel: 'Requests per Month',
  utilizationBreakdownLabel: 'Growth Factor',
  commissionLabel: 'Self-hosted Cost',
  yourTakeHomeLabel: 'Your Net Savings',
  ctaLabel: 'Start Saving Now',
  ctaAriaLabel: 'Get started with rbee to start saving',
  secondaryCTALabel: 'See detailed comparison',
  formatCurrency: (n, opts) => `$${n.toFixed(opts?.maximumFractionDigits ?? 0)}`,
  formatHourly: (n) => `$${n.toFixed(3)}`,
}

// === Growth Roadmap (adapted EnterpriseHowItWorks) ===
export const startupsGrowthRoadmapContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Your Growth Path',
  title: 'From MVP to Scale: Own Your Infrastructure at Every Stage',
  description: 'Start small, scale big. rbee grows with you—without the API bills.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsGrowthRoadmapProps: EnterpriseHowItWorksProps = {
  deploymentSteps: [
    {
      index: 1,
      icon: <Rocket className="h-6 w-6" />,
      title: 'MVP Stage',
      intro: 'Prototype fast with full control. Start with one GPU and build without worrying about API costs.',
      items: ['Single GPU setup', 'OpenAI-compatible API', '$0 per token'],
    },
    {
      index: 2,
      icon: <Server className="h-6 w-6" />,
      title: 'Launch Stage',
      intro: 'Scale to production. Add more GPUs as you get traction.',
      items: ['Multi-GPU orchestration', 'Unlimited requests', 'Production-ready'],
    },
    {
      index: 3,
      icon: <BarChart3 className="h-6 w-6" />,
      title: 'Scale Stage',
      intro: 'Grow without limits. Distribute across multiple machines.',
      items: ['Distributed cluster', 'Enterprise scale', 'Predictable costs'],
    },
  ],
  timeline: {
    heading: 'Growth Timeline',
    description: 'From MVP to scale',
    weeks: [
      { week: 'Week 1', phase: 'MVP Launch' },
      { week: 'Week 4', phase: 'First Customers' },
      { week: 'Week 12', phase: 'Production Scale' },
      { week: 'Week 24', phase: 'Enterprise Ready' },
    ],
  },
}

// === Startup Scenarios (UseCasesTemplate) ===
export const startupsUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Built for Startups',
  title: 'Real Startups, Real Savings',
  description: 'See how different types of startups use rbee to build AI products without burning cash.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsUseCasesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <Code className="h-6 w-6" />,
      title: 'B2B SaaS',
      scenario: 'Building an AI-powered code review tool. API costs were $3K/mo and growing fast.',
      solution: 'Switched to rbee. Self-hosted on 3 GPUs. Same performance, zero per-token costs.',
      outcome: '$36K saved in year one. Reinvested in product features.',
      tags: ['Code Analysis', 'Developer Tools'],
    },
    {
      icon: <Users className="h-6 w-6" />,
      title: 'Consumer App',
      scenario: "Chat app with 10K users. OpenAI bills hit $5K/mo. Couldn't afford to scale.",
      solution: 'Deployed rbee on rented GPUs. Cut costs by 85%. Scaled to 50K users.',
      outcome: 'Profitable within 6 months. Raised Series A.',
      tags: ['Chat', 'Consumer'],
    },
    {
      icon: <Rocket className="h-6 w-6" />,
      title: 'AI-First Startup',
      scenario: 'Building AI agents. Needed multiple models. API costs were unpredictable.',
      solution: 'rbee cluster with 8 GPUs. Run Llama, Mistral, and custom models simultaneously.',
      outcome: 'Full control over models and costs. Shipped 3x faster.',
      tags: ['AI Agents', 'Multi-Model'],
    },
  ],
  columns: 3,
}

// === Comparison Template ===
export const startupsComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Side by Side',
  title: 'rbee vs API Providers',
  description: 'See why startups choose self-hosted over APIs.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsComparisonProps: ComparisonTemplateProps = {
  columns: [
    { key: 'rbee', label: 'rbee (Self-Hosted)', accent: true },
    { key: 'openai', label: 'OpenAI API' },
    { key: 'anthropic', label: 'Anthropic' },
  ],
  rows: [
    {
      feature: 'Monthly Cost (1M tokens)',
      values: {
        rbee: '$0',
        openai: '$2,400',
        anthropic: '$2,000',
      },
    },
    {
      feature: 'Rate Limits',
      values: {
        rbee: true,
        openai: false,
        anthropic: false,
      },
    },
    {
      feature: 'Data Privacy',
      values: {
        rbee: '100% Private',
        openai: 'Sent to OpenAI',
        anthropic: 'Sent to Anthropic',
      },
    },
    {
      feature: 'Model Control',
      values: {
        rbee: 'Full Control',
        openai: 'Limited',
        anthropic: 'Limited',
      },
    },
    {
      feature: 'Vendor Lock-In',
      values: {
        rbee: true,
        openai: false,
        anthropic: false,
      },
    },
    {
      feature: 'OpenAI-Compatible',
      values: {
        rbee: true,
        openai: true,
        anthropic: false,
      },
    },
  ],
  legend: [
    { icon: <Check className="h-3.5 w-3.5 text-emerald-500" />, label: 'Available / No Lock-In' },
    { icon: <X className="h-3.5 w-3.5 text-destructive" />, label: 'Not Available / Locked In' },
  ],
  showMobileCards: true,
}

// === Technical Stack ===
export const startupsTechnicalContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Built for Developers',
  title: 'Drop-In Replacement for OpenAI',
  description: 'Change one line of code. Your existing app works unchanged.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsTechnicalProps: TechnicalTemplateProps = {
  architectureHighlights: [
    {
      title: 'OpenAI-Compatible',
      details: ['Drop-in replacement', 'No code changes'],
    },
    {
      title: 'All GPUs',
      details: ['CUDA, Metal, CPU', 'Works anywhere'],
    },
    {
      title: 'Multi-Model',
      details: ['Llama, Mistral, Qwen', 'Run any model'],
    },
  ],
  techStack: [
    {
      name: 'OpenAI SDK',
      description: 'Drop-in compatible API',
      ariaLabel: 'Tech: OpenAI SDK compatibility',
    },
    {
      name: 'Multi-Backend',
      description: 'CUDA, Metal, CPU support',
      ariaLabel: 'Tech: Multi-backend GPU support',
    },
    {
      name: 'Model Catalog',
      description: 'Llama, Mistral, Qwen, and more',
      ariaLabel: 'Tech: Model catalog',
    },
  ],
  stackLinks: {
    githubUrl: 'https://github.com/veighnsche/llama-orch',
    license: 'GPL-3.0-or-later',
    architectureUrl: '/technical-deep-dive',
  },
}

// === Testimonials ===
export const startupsTestimonialsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Founder Stories',
  title: 'Startups Building on rbee',
  description: 'Hear from founders who escaped the API cost trap.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const startupsTestimonialsProps: TestimonialsTemplateProps = {
  testimonials: [
    {
      quote:
        'We were burning $4K/mo on OpenAI. Switched to rbee and cut that to $400 in hardware costs. Same performance, 90% savings.',
      author: 'Sarah Chen',
      role: 'Founder, DevTools Startup',
      avatar: '👩‍💻',
    },
    {
      quote:
        'Rate limits were killing our growth. With rbee, we handle 10x the traffic with zero throttling. Total game changer.',
      author: 'Marcus Johnson',
      role: 'CTO, AI Chat App',
      avatar: '👨‍💼',
    },
    {
      quote:
        "Being able to run multiple models simultaneously gave us a huge competitive advantage. API providers couldn't match our speed.",
      author: 'Elena Rodriguez',
      role: 'CEO, AI Agent Platform',
      avatar: '👩‍💼',
    },
  ],
}

// === FAQ ===
export const startupsFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Common Questions',
  title: 'Startup FAQs',
  description: 'Everything you need to know about using rbee for your startup.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
}

export const startupsFAQProps: FAQTemplateProps = {
  categories: ['General', 'Technical', 'Pricing'],
  faqItems: [
    {
      value: 'cost',
      question: 'How much does it really cost to self-host?',
      answer:
        'Hardware costs vary. A single RTX 4090 (~$1,600) can handle most MVP workloads. Compare that to $2,400/mo in API fees—you break even in under a month. As you scale, add more GPUs. No recurring per-token costs.',
      category: 'Pricing',
    },
    {
      value: 'compatible',
      question: 'Is it really OpenAI-compatible?',
      answer:
        "Yes. rbee implements the OpenAI API spec. Change your baseURL and you're done. Your existing code, SDKs, and tools work unchanged. We support chat completions, embeddings, and streaming.",
      category: 'Technical',
    },
    {
      value: 'no-gpus',
      question: "What if I don't have GPUs?",
      answer:
        'Rent them. Providers like Vast.ai, RunPod, and Lambda Labs offer GPU rentals starting at $0.20/hr. Still 80-90% cheaper than API providers, with no rate limits.',
      category: 'General',
    },
    {
      value: 'switch-back',
      question: 'Can I switch back to OpenAI if needed?',
      answer:
        'Absolutely. Since rbee is OpenAI-compatible, switching back is just changing the baseURL. No vendor lock-in. You own your infrastructure and your choices.',
      category: 'General',
    },
    {
      value: 'setup-time',
      question: 'How long does setup take?',
      answer:
        "About 10 minutes. Install rbee, point your code to localhost, and you're running. Our quickstart guide walks you through everything step by step.",
      category: 'Technical',
    },
    {
      value: 'support',
      question: 'What about support and maintenance?',
      answer:
        'rbee is open source with active community support. For startups needing SLAs, we offer paid support plans starting at $500/mo—still far cheaper than API costs.',
      category: 'Pricing',
    },
  ],
}

// === CTA Template ===
export const startupsCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'gradient-primary',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
}

export const startupsCTAProps: CTATemplateProps = {
  eyebrow: 'Ready to Own Your Stack?',
  title: 'Start Building Without API Fees',
  subtitle: 'Join 500+ startups using rbee to build AI products on their own terms.',
  primary: {
    label: 'Get Started Free',
    href: '/getting-started',
    iconRight: ArrowRight,
  },
  secondary: {
    label: 'Talk to Founders',
    href: '/community',
  },
  note: '100% open source • GPL-3.0-or-later • No vendor lock-in',
  emphasis: 'gradient',
}
