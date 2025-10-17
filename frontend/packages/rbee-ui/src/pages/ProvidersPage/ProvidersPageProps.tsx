'use client'

import { gpuEarnings } from '@rbee/ui/assets'
import { TESTIMONIALS } from '@rbee/ui/data/testimonials'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { CommissionStructureCard, ProvidersSecurityCard } from '@rbee/ui/molecules'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import type { CommissionStructureCardProps } from '@rbee/ui/molecules/CommissionStructureCard'
import type { ProvidersSecurityCardProps } from '@rbee/ui/molecules/ProvidersSecurityCard'
import { ProvidersCaseCard } from '@rbee/ui/organisms'
import type { ProvidersCaseCardProps } from '@rbee/ui/organisms/ProvidersCaseCard'
import type {
  CardGridTemplateProps,
  FeaturesTabsProps,
  HowItWorksProps,
  ProvidersCTAProps,
  ProvidersEarningsProps,
  ProvidersHeroProps,
  TestimonialsTemplateProps,
} from '@rbee/ui/templates'
import type { ProblemTemplateProps } from '@rbee/ui/templates/ProblemTemplate'
import type { SolutionTemplateProps } from '@rbee/ui/templates/SolutionTemplate'
import {
  AlertCircle,
  BarChart3,
  Clock,
  Cpu,
  DollarSign,
  Eye,
  FileCheck,
  Gamepad2,
  Globe,
  Lock,
  Monitor,
  Server,
  Shield,
  Sliders,
  Star,
  TrendingDown,
  TrendingUp,
  Users,
  Wallet,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects
// ============================================================================

// === ProvidersHero Template ===
export const providersHeroProps: ProvidersHeroProps = {
  kickerIcon: <Zap className="h-3.5 w-3.5" />,
  kickerText: '💡 Turn Idle GPUs Into Income',
  headline: 'Your GPUs Can Pay You Every Month',
  supportingText:
    'Join the rbee marketplace and get paid when developers use your spare compute. Plug in once, set your price, and start earning automatically.',
  stats: [
    {
      icon: <DollarSign className="w-6 h-6" />,
      value: '€50–200',
      label: 'per GPU / month',
    },
    {
      icon: <Clock className="w-6 h-6" />,
      value: '24/7',
      label: 'Passive income',
    },
    {
      icon: <Shield className="w-6 h-6" />,
      value: '100%',
      label: 'Secure payouts',
    },
  ],
  primaryCTA: {
    label: 'Start Earning',
    ariaLabel: 'Start earning with rbee',
  },
  secondaryCTA: {
    label: 'Estimate My Payout',
    href: '#earnings-calculator',
  },
  trustLine: 'No tech expertise needed • Set your own prices • Pause anytime',
  dashboard: {
    icon: <BarChart3 className="w-6 h-6" />,
    title: 'Your Earnings Dashboard',
    statusBadge: 'Active',
    monthLabel: 'This Month',
    monthEarnings: '€156.80',
    monthGrowth: '+23%',
    progressPercentage: 56,
    totalHours: '487',
    avgRate: '€0.32/hr',
    gpuListTitle: 'Your GPUs',
    gpus: [
      {
        name: 'RTX 4090',
        location: 'Gaming PC',
        earnings: '€89.20',
        status: 'active',
      },
      {
        name: 'RTX 3080',
        location: 'Workstation',
        earnings: '€67.60',
        status: 'active',
      },
    ],
  },
}

// === ProvidersProblem Template ===
export const providersProblemContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'The Cost of Idle GPUs',
  title: 'Stop Letting Your Hardware Bleed Money',
  description: 'Most GPUs sit idle ~90% of the time. They still draw power—and earn nothing.',
  bgVariant: 'destructive-gradient',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctaBanner: {
    copy: 'Every idle hour is money left on the table. Turn that waste into passive income with rbee.',
    primary: { label: 'Start Earning', href: '/signup' },
    secondary: { label: 'Estimate My Payout', href: '#earnings-calculator' },
  },
}

export const providersProblemProps: ProblemTemplateProps = {
  items: [
    {
      icon: <TrendingDown className="h-6 w-6" />,
      title: 'Wasted Investment',
      body: "You paid €1,500+ for a high-end GPU. It's busy maybe 10% of the time—the other 90% earns €0.",
      tag: 'Potential earnings €50-200/mo',
      tone: 'destructive',
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: 'Electricity Costs',
      body: "Idle GPUs still pull power. That's roughly €10-30 each month spent on doing nothing.",
      tag: 'Direct loss €10-30/mo',
      tone: 'destructive',
    },
    {
      icon: <AlertCircle className="h-6 w-6" />,
      title: 'Missed Opportunity',
      body: 'Developers rent GPU power every day. Your machine could join the marketplace and get paid automatically.',
      tag: 'Unrealized €50-200/mo',
      tone: 'destructive',
    },
  ],
}

// === ProvidersSolution Template ===
export const providersSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'How rbee Works',
  title: 'Turn Idle GPUs Into Reliable Monthly Income',
  description: 'rbee connects your spare GPU power to developers who need it. You set the price, we handle the rest.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctas: {
    primary: {
      label: 'Start Earning',
      href: '/signup',
      ariaLabel: 'Start earning with rbee',
    },
    secondary: {
      label: 'Estimate My Payout',
      href: '#earnings-calculator',
    },
  },
}

export const providersSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <DollarSign className="h-8 w-8" aria-hidden="true" />,
      title: 'Passive Income',
      body: 'Earn €50–200/mo per GPU—even while you game or sleep.',
    },
    {
      icon: <Sliders className="h-8 w-8" aria-hidden="true" />,
      title: 'Full Control',
      body: 'Set prices, availability windows, and usage limits.',
    },
    {
      icon: <Shield className="h-8 w-8" aria-hidden="true" />,
      title: 'Secure & Private',
      body: 'Sandboxed jobs. No access to your files.',
    },
    {
      icon: <Zap className="h-8 w-8" aria-hidden="true" />,
      title: 'Easy Setup',
      body: 'Install in ~10 minutes. No expertise required.',
    },
  ],
  steps: [
    {
      title: 'Install rbee',
      body: 'Run one command on Windows, macOS, or Linux.',
    },
    {
      title: 'Configure Your GPUs',
      body: 'Choose pricing, availability, and usage limits in the web dashboard.',
    },
    {
      title: 'Join the Marketplace',
      body: 'Your GPUs become rentable to verified developers.',
    },
    {
      title: 'Get Paid',
      body: 'Earnings track in real time. Withdraw anytime.',
    },
  ],
  earnings: {
    rows: [
      {
        model: 'RTX 4090',
        meta: '24GB VRAM • 450W',
        value: '€180/mo',
        note: 'at 80% utilization',
      },
      {
        model: 'RTX 4080',
        meta: '16GB VRAM • 320W',
        value: '€140/mo',
        note: 'at 80% utilization',
      },
      {
        model: 'RTX 3080',
        meta: '10GB VRAM • 320W',
        value: '€90/mo',
        note: 'at 80% utilization',
      },
    ],
    disclaimer: 'Actuals vary with demand, pricing, and availability. These are conservative estimates.',
  },
}

// === ProvidersHowItWorks Template ===

/**
 * Providers How It Works container - Layout configuration
 */
export const providersHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Start Earning in 4 Simple Steps',
  bgVariant: 'secondary',
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Providers How It Works - Step-by-step guide to start earning
 */
export const providersHowItWorksProps: HowItWorksProps = {
  steps: [
    {
      label: 'Install rbee',
      block: {
        kind: 'terminal',
        title: 'Installation',
        lines: <>curl -sSL rbee.dev/install.sh | sh</>,
        copyText: 'curl -sSL rbee.dev/install.sh | sh',
      },
    },
    {
      label: 'Configure Settings',
      block: {
        kind: 'note',
        content: (
          <>
            Set your pricing, availability windows, and usage limits through the intuitive web dashboard.
            <ul className="mt-2 space-y-1 text-sm">
              <li>• Set hourly rate</li>
              <li>• Define availability</li>
              <li>• Set usage limits</li>
            </ul>
          </>
        ),
      },
    },
    {
      label: 'Join Marketplace',
      block: {
        kind: 'note',
        content: (
          <>
            Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute power.
            <div className="mt-2 text-sm font-medium text-green-600">✓ Your GPUs are now live and earning.</div>
          </>
        ),
      },
    },
    {
      label: 'Get Paid',
      block: {
        kind: 'note',
        content: (
          <>
            Track earnings in real time. Automatic payouts to your bank or crypto wallet.
            <div className="mt-2 space-y-1 text-sm">
              <div>
                <strong>Payout frequency:</strong> Weekly
              </div>
              <div>
                <strong>Minimum payout:</strong> €25
              </div>
            </div>
          </>
        ),
      },
    },
  ],
}

// === ProvidersFeatures Template ===
export const providersFeaturesProps: FeaturesTabsProps = {
  title: 'Everything You Need to Maximize Earnings',
  description: 'Professional-grade tools to manage your GPU fleet and optimize your passive income.',
  tabs: [
    {
      value: 'pricing',
      icon: <DollarSign className="size-4" />,
      label: 'Flexible Pricing Control',
      mobileLabel: 'Pricing',
      subtitle: 'Set your rates',
      badge: 'Control',
      description: 'Set your own hourly rates based on GPU model, demand, and your preferences.',
      content: (
        <CodeBlock
          code={`{
  "gpu": "RTX 4090",
  "base_rate": 1.50,
  "min_rate": 1.00,
  "max_rate": 3.00,
  "demand_multiplier": true,
  "schedule": {
    "weekday": 1.5,
    "weekend": 2.0
  }
}`}
          language="json"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Dynamic pricing based on demand with automatic adjustments',
        variant: 'primary',
      },
      benefits: [{ text: 'Set custom rates' }, { text: 'Demand-based pricing' }, { text: 'Schedule multipliers' }],
    },
    {
      value: 'availability',
      icon: <Clock className="size-4" />,
      label: 'Availability Management',
      mobileLabel: 'Schedule',
      subtitle: 'Control when',
      badge: 'Flexible',
      description: 'Control exactly when your GPUs are available for rent.',
      content: (
        <CodeBlock
          code={`{
  "weekday": "09:00-17:00",
  "weekend": "all-day",
  "vacation_mode": false,
  "priority_mode": "my_usage_first",
  "auto_pause_gaming": true
}`}
          language="json"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Set availability windows and priority modes',
        variant: 'primary',
      },
      benefits: [{ text: 'Custom schedules' }, { text: 'Priority modes' }, { text: 'Auto-pause gaming' }],
    },
    {
      value: 'security',
      icon: <Shield className="size-4" />,
      label: 'Security & Privacy',
      mobileLabel: 'Security',
      subtitle: 'Protected',
      badge: 'Safe',
      description: 'Your data and hardware are protected with enterprise-grade security.',
      content: (
        <div className="space-y-2 text-sm">
          <div>✓ Sandboxed execution (no file access)</div>
          <div>✓ Encrypted communication (TLS 1.3)</div>
          <div>✓ No access to personal data</div>
          <div>✓ Malware scanning on all jobs</div>
          <div>✓ Automatic security updates</div>
          <div>✓ Insurance coverage included</div>
        </div>
      ),
      highlight: {
        text: 'Sandboxed execution with encrypted communication',
        variant: 'success',
      },
      benefits: [{ text: 'Sandboxed jobs' }, { text: 'TLS encryption' }, { text: 'Insurance included' }],
    },
    {
      value: 'analytics',
      icon: <BarChart3 className="size-4" />,
      label: 'Earnings Dashboard',
      mobileLabel: 'Analytics',
      subtitle: 'Track earnings',
      badge: 'Insights',
      description: 'Track your earnings, utilization, and performance in real‑time.',
      content: (
        <div className="space-y-2 font-mono text-sm">
          <div>
            <strong>Today:</strong> €42.50
          </div>
          <div>
            <strong>This Week:</strong> €287.30
          </div>
          <div>
            <strong>This Month:</strong> €1,124.80
          </div>
          <div className="mt-4 space-y-1 border-t border-border pt-2">
            <div>
              <strong>Utilization:</strong> 78%
            </div>
            <div>
              <strong>Avg Rate:</strong> €1.85/hr
            </div>
            <div>
              <strong>Top GPU:</strong> RTX 4090 (€524/mo)
            </div>
          </div>
        </div>
      ),
      highlight: {
        text: 'Real‑time earnings tracking with historical charts',
        variant: 'primary',
      },
      benefits: [{ text: 'Real-time tracking' }, { text: 'Historical charts' }, { text: 'Performance metrics' }],
    },
  ],
  defaultTab: 'pricing',
}

/**
 * Features tabs container - Background wrapper
 */
export const providersFeaturesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
}

// === ProvidersUseCases Template ===
export const providersUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Real Providers, Real Earnings',
  title: "Who's Earning with rbee?",
  description: 'From gamers to homelab builders, anyone with a spare GPU can turn idle time into income.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const providersUseCasesProps: { cases: ProvidersCaseCardProps[] } = {
  cases: [
    {
      icon: <Gamepad2 />,
      title: 'Gaming PC Owners',
      subtitle: 'Most common provider type',
      quote: "I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep.",
      facts: [
        { label: 'Typical GPU:', value: 'RTX 4080–4090' },
        { label: 'Availability:', value: '16–20 h/day' },
        { label: 'Monthly:', value: '€120–180' },
      ],
    },
    {
      icon: <Server />,
      title: 'Homelab Enthusiasts',
      subtitle: 'Multiple GPUs, high earnings',
      quote: 'Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit.',
      facts: [
        { label: 'Setup:', value: '3–6 GPUs' },
        { label: 'Availability:', value: '20–24 h/day' },
        { label: 'Monthly:', value: '€300–600' },
      ],
    },
    {
      icon: <Cpu />,
      title: 'Former Crypto Miners',
      subtitle: 'Repurpose mining rigs',
      quote: 'After PoS, my rig idled. rbee now earns more than mining—with better margins.',
      facts: [
        { label: 'Setup:', value: '6–12 GPUs' },
        { label: 'Availability:', value: '24 h/day' },
        { label: 'Monthly:', value: '€600–1,200' },
      ],
    },
    {
      icon: <Monitor />,
      title: 'Workstation Owners',
      subtitle: 'Professional GPUs earning',
      quote: 'My RTX 4080 is busy on renders only. The rest of the time it makes ~€100/mo on rbee.',
      facts: [
        { label: 'Typical GPU:', value: 'RTX 4070–4080' },
        { label: 'Availability:', value: '12–16 h/day' },
        { label: 'Monthly:', value: '€80–140' },
      ],
    },
  ],
}

export const providersUseCasesGridProps: CardGridTemplateProps = {
  columns: 2,
  gap: 'md',
  cards: providersUseCasesProps.cases.map((caseData, index) => (
    <ProvidersCaseCard
      key={index}
      icon={caseData.icon}
      title={caseData.title}
      subtitle={caseData.subtitle}
      quote={caseData.quote}
      facts={caseData.facts}
      highlight={caseData.highlight}
      index={index}
    />
  )),
}

// === ProvidersEarnings Template ===
const fmt = (n: number, opts: Intl.NumberFormatOptions = {}) =>
  new Intl.NumberFormat('en-IE', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
    ...opts,
  }).format(n)

const fmtHr = (n: number) => `${fmt(n, { maximumFractionDigits: 2 })}/hr`

export const providersEarningsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Estimate Your Earnings',
  title: 'Calculate Your Potential Earnings',
  description: 'See what you could earn based on GPU model, availability, and utilization.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '4xl',
  align: 'center',
}

export const providersEarningsProps: ProvidersEarningsProps = {
  gpuModels: [
    { name: 'RTX 4090', baseRate: 0.45, vram: 24 },
    { name: 'RTX 4080', baseRate: 0.35, vram: 16 },
    { name: 'RTX 4070 Ti', baseRate: 0.28, vram: 12 },
    { name: 'RTX 3090', baseRate: 0.32, vram: 24 },
    { name: 'RTX 3080', baseRate: 0.25, vram: 10 },
    { name: 'RTX 3070', baseRate: 0.18, vram: 8 },
  ],
  presets: [
    { label: 'Casual', hours: 8, utilization: 50 },
    { label: 'Daily', hours: 16, utilization: 70 },
    { label: 'Always On', hours: 24, utilization: 90 },
  ],
  commission: 0.15,
  configTitle: 'Your Configuration',
  selectGPULabel: 'Select Your GPU',
  presetsLabel: 'Quick Presets',
  hoursLabel: 'Hours Available Per Day',
  utilizationLabel: 'Expected Utilization',
  earningsTitle: 'Your Potential Earnings',
  monthlyLabel: 'Monthly Earnings',
  basedOnText: (hours: number, utilization: number) => `Based on ${hours}h/day at ${utilization}% utilization`,
  takeHomeLabel: 'Take-home (after 15%)',
  dailyLabel: 'Daily',
  yearlyLabel: 'Yearly',
  breakdownTitle: 'Breakdown',
  hourlyRateLabel: 'Hourly rate',
  hoursPerMonthLabel: 'Hours per month',
  utilizationBreakdownLabel: 'Utilization',
  commissionLabel: 'rbee commission (15%)',
  yourTakeHomeLabel: 'Your take-home',
  ctaLabel: 'Start Earning Now',
  ctaAriaLabel: 'Start earning with rbee',
  secondaryCTALabel: 'Estimate on another GPU',
  formatCurrency: fmt,
  formatHourly: fmtHr,
}

// === ProvidersMarketplace Template ===
export const providersMarketplaceContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Why rbee',
  title: 'How the rbee Marketplace Works',
  description: 'A fair, transparent marketplace connecting GPU providers with developers.',
  bgVariant: 'secondary',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const providersMarketplaceCommissionProps: CommissionStructureCardProps = {
  title: 'Commission Structure',
  standardCommissionLabel: 'Standard Commission',
  standardCommissionValue: '15%',
  standardCommissionDescription: 'Covers marketplace operations, payouts, and support.',
  youKeepLabel: 'You Keep',
  youKeepValue: '85%',
  youKeepDescription: 'No hidden fees or surprise deductions.',
  exampleItems: [
    { label: 'Example job', value: '€100.00' },
    { label: 'rbee commission (15%)', value: '−€15.00' },
  ],
  exampleTotalLabel: 'Your earnings',
  exampleTotalValue: '€85.00',
  exampleBadgeText: 'Effective take-home: 85%',
}

export const providersMarketplaceSolutionProps: SolutionTemplateProps = {
  aside: (
    <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
      <CommissionStructureCard {...providersMarketplaceCommissionProps} />
    </div>
  ),
  features: [
    {
      icon: <TrendingUp className="size-6" />,
      title: 'Dynamic Pricing',
      body: 'Set your own rate or use auto-pricing.',
    },
    {
      icon: <Users className="size-6" />,
      title: 'Growing Demand',
      body: 'Thousands of AI jobs posted monthly.',
    },
    {
      icon: <Globe className="size-6" />,
      title: 'Global Reach',
      body: 'Your GPUs are discoverable worldwide.',
    },
    {
      icon: <Shield className="size-6" />,
      title: 'Fair Commission',
      body: 'Keep 85% of every payout.',
    },
  ],
  steps: [
    {
      title: 'Automatic Matching',
      body: 'Jobs match your GPUs based on specs and your pricing.',
    },
    {
      title: 'Rating System',
      body: 'Higher ratings unlock more jobs and better rates.',
    },
    {
      title: 'Guaranteed Payments',
      body: 'Customers pre-pay. Every completed job is paid.',
    },
    {
      title: 'Dispute Resolution',
      body: 'A fair process protects both providers and customers.',
    },
  ],
}

// === ProvidersSecurity Template ===
export const providersSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Security & Trust',
  title: 'Your Security Is Our Priority',
  description: 'Enterprise-grade protections for your hardware, data, and earnings.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '4xl',
  align: 'center',
}

export const providersSecurityProps: { items: ProvidersSecurityCardProps[] } = {
  items: [
    {
      icon: <Shield className="size-6" />,
      title: 'Sandboxed Execution',
      subtitle: 'Complete isolation',
      body: 'All jobs run in isolated sandboxes with no access to your files, network, or personal data.',
      points: ['No file system access', 'No network access', 'No personal data access', 'Automatic cleanup'],
    },
    {
      icon: <Lock className="size-6" />,
      title: 'Encrypted Communication',
      subtitle: 'End-to-end encryption',
      body: 'All communication between your GPU and the marketplace is encrypted using industry-standard protocols.',
      points: ['TLS 1.3', 'Secure payment processing', 'Protected earnings data', 'Private job details'],
    },
    {
      icon: <Eye className="size-6" />,
      title: 'Malware Scanning',
      subtitle: 'Automatic protection',
      body: 'Every job is automatically scanned for malware before execution. Suspicious jobs are blocked.',
      points: ['Real-time detection', 'Automatic blocking', 'Threat intel updates', 'Customer vetting'],
    },
    {
      icon: <FileCheck className="size-6" />,
      title: 'Hardware Protection',
      subtitle: 'Warranty-safe operation',
      body: 'Temperature monitoring, cooldown periods, and power limits protect your hardware and warranty.',
      points: ['Warranty-safe operation', 'Temperature monitoring', 'Cooldown periods', 'Power limit controls'],
    },
  ],
}

export const providersSecurityGridProps: CardGridTemplateProps = {
  columns: 2,
  gap: 'md',
  cards: providersSecurityProps.items.map((item, index) => (
    <ProvidersSecurityCard
      key={index}
      icon={item.icon}
      title={item.title}
      subtitle={item.subtitle}
      body={item.body}
      points={item.points}
      index={index}
    />
  )),
}

// === ProvidersTestimonials Template ===
export const providersTestimonialsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Provider Stories',
  title: 'What Real Providers Are Earning',
  description:
    'GPU owners on the rbee marketplace turn idle time into steady payouts—fully self-managed, OpenAI-compatible infrastructure.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const providersTestimonialsProps: TestimonialsTemplateProps = {
  testimonials: TESTIMONIALS.filter((t) => t.sector === 'provider').map((t) => ({
    quote: t.quote,
    author: t.name,
    role: t.role,
    avatar: t.avatar,
  })),
  stats: [
    {
      icon: <Users className="w-6 h-6" />,
      value: '500+',
      label: 'Active Providers',
    },
    {
      icon: <Cpu className="w-6 h-6" />,
      value: '2,000+',
      label: 'GPUs Earning',
    },
    {
      icon: <TrendingUp className="w-6 h-6" />,
      value: '€180K+',
      label: 'Paid to Providers',
    },
    {
      icon: <Star className="w-6 h-6" />,
      value: '4.8/5',
      label: 'Average Rating',
    },
  ],
}

// === ProvidersCTA Template ===
export const providersCTAProps: ProvidersCTAProps = {
  badgeIcon: <Zap className="h-4 w-4" aria-hidden="true" />,
  badgeText: 'Start earning today',
  title: 'Turn Idle GPUs Into Weekly Payouts',
  subtitle: 'Join 500+ providers monetizing spare GPU time on the rbee marketplace.',
  primaryCTA: {
    label: 'Start Earning Now',
    ariaLabel: 'Start earning now — setup under 15 minutes',
  },
  secondaryCTA: {
    label: 'View Docs',
    ariaLabel: 'View documentation for providers',
  },
  disclaimerText: 'Data from verified providers; earnings vary by GPU, uptime, and demand.',
  stats: [
    {
      icon: <Clock className="w-6 h-6" />,
      value: '< 15 minutes',
      label: 'Setup time',
    },
    {
      icon: <Shield className="w-6 h-6" />,
      value: '15% platform fee',
      label: 'You keep 85%',
    },
    {
      icon: <Wallet className="w-6 h-6" />,
      value: '€25 minimum',
      label: 'Weekly payouts',
    },
  ],
  backgroundImage: {
    src: gpuEarnings,
    alt: 'Cinematic macro shot of three modern NVIDIA RTX GPUs stacked vertically with visible cooling fans and RGB accents, emitting warm amber and orange volumetric light rays from their edges; translucent holographic euro currency symbols (€) and AI task tokens with neural network patterns float upward in a gentle arc representing passive income; dark navy blue to black gradient backdrop with subtle hexagonal mesh pattern; shallow depth of field with bokeh effect; dramatic side lighting creating rim light on GPU edges; photorealistic 3D render style; high contrast with deep shadows; premium tech aesthetic; 8K quality; particles of light dust in the air catching the amber glow; emphasis on AI workload monetization not cryptocurrency mining',
  },
}
