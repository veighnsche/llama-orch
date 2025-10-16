'use client'

import { gpuEarnings } from '@rbee/ui/assets'
import { FormerCryptoMiner, GamingPcOwner, HomelabEnthusiast, WorkstationOwner } from '@rbee/ui/icons'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type { FeatureTabsSectionProps, StepsSectionProps } from '@rbee/ui/organisms'
import type {
  ProvidersCTATemplateProps,
  ProvidersEarningsTemplateProps,
  ProvidersHeroTemplateProps,
  ProvidersMarketplaceTemplateProps,
  ProvidersSecurityTemplateProps,
  ProvidersTestimonialsTemplateProps,
  ProvidersUseCasesTemplateProps,
} from '@rbee/ui/templates'
import type { ProblemTemplateProps } from '@rbee/ui/templates/ProblemTemplate'
import type { SolutionTemplateProps } from '@rbee/ui/templates/SolutionTemplate'
import {
  AlertCircle,
  BarChart3,
  Clock,
  Cpu,
  DollarSign,
  Download,
  Eye,
  FileCheck,
  Gamepad2,
  Globe,
  Lock,
  Monitor,
  Server,
  Settings,
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
export const providersHeroProps: ProvidersHeroTemplateProps = {
  kickerIcon: <Zap className="h-3.5 w-3.5" />,
  kickerText: '💡 Turn Idle GPUs Into Income',
  headline: 'Your GPUs Can Pay You Every Month',
  supportingText:
    'Join the rbee marketplace and get paid when developers use your spare compute. Plug in once, set your price, and start earning automatically.',
  stats: [
    {
      icon: DollarSign,
      value: '€50–200',
      label: 'per GPU / month',
    },
    {
      icon: Clock,
      value: '24/7',
      label: 'Passive income',
    },
    {
      icon: Shield,
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
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
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
  ctaPrimary: { label: 'Start Earning', href: '/signup' },
  ctaSecondary: { label: 'Estimate My Payout', href: '#earnings-calculator' },
  ctaCopy: 'Every idle hour is money left on the table. Turn that waste into passive income with rbee.',
}

// === ProvidersSolution Template ===
export const providersSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'How rbee Works',
  title: 'Turn Idle GPUs Into Reliable Monthly Income',
  description:
    'rbee connects your GPUs with developers who need compute. You set the price, control availability, and get paid automatically.',
  bgVariant: 'secondary',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
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
  ctaPrimary: {
    label: 'Start Earning',
    href: '/signup',
    ariaLabel: 'Start earning with rbee',
  },
  ctaSecondary: {
    label: 'Estimate My Payout',
    href: '#earnings-calculator',
  },
}

// === ProvidersHowItWorks Template ===
// Note: Using StepsSection organism
export const providersHowItWorksProps: StepsSectionProps = {
  id: 'how-it-works',
  kicker: 'How rbee Works',
  title: 'Start Earning in 4 Simple Steps',
  subtitle: 'No technical expertise required. Most providers finish in ~15 minutes.',
  steps: [
    {
      icon: <Download className="h-8 w-8" aria-hidden="true" />,
      step: 'Step 1',
      title: 'Install rbee',
      body: 'Download and install with one command. Works on Windows, macOS, and Linux.',
      snippet: 'curl -sSL rbee.dev/install.sh | sh',
    },
    {
      icon: <Settings className="h-8 w-8" aria-hidden="true" />,
      step: 'Step 2',
      title: 'Configure Settings',
      body: 'Set your pricing, availability windows, and usage limits through the intuitive web dashboard.',
      checklist: ['Set hourly rate', 'Define availability', 'Set usage limits'],
    },
    {
      icon: <Globe className="h-8 w-8" aria-hidden="true" />,
      step: 'Step 3',
      title: 'Join Marketplace',
      body: 'Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute power.',
      successNote: 'Your GPUs are now live and earning.',
    },
    {
      icon: <Wallet className="h-8 w-8" aria-hidden="true" />,
      step: 'Step 4',
      title: 'Get Paid',
      body: 'Track earnings in real time. Automatic payouts to your bank or crypto wallet.',
      stats: [
        { label: 'Payout frequency', value: 'Weekly' },
        { label: 'Minimum payout', value: '€25' },
      ],
    },
  ],
  avgTime: '12 minutes',
}

// === ProvidersFeatures Template ===
// Note: Using FeatureTabsSection organism
export const providersFeaturesProps: FeatureTabsSectionProps = {
  title: 'Everything You Need to Maximize Earnings',
  subtitle: 'Professional-grade tools to manage your GPU fleet and optimize your passive income.',
  items: [
    {
      id: 'pricing',
      title: 'Flexible Pricing Control',
      description: 'Set your own hourly rates based on GPU model, demand, and your preferences.',
      icon: <DollarSign className="h-4 w-4" />,
      benefit: { text: 'Dynamic pricing based on demand with automatic adjustments', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Pricing Configuration',
        content: `{
  "gpu": "RTX 4090",
  "base_rate": 1.50,
  "min_rate": 1.00,
  "max_rate": 3.00,
  "demand_multiplier": true,
  "schedule": {
    "weekday": 1.5,
    "weekend": 2.0
  }
}`,
      },
    },
    {
      id: 'availability',
      title: 'Availability Management',
      description: 'Control exactly when your GPUs are available for rent.',
      icon: <Clock className="h-4 w-4" />,
      benefit: { text: 'Set availability windows and priority modes', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Availability Schedule',
        content: `{
  "weekday": "09:00-17:00",
  "weekend": "all-day",
  "vacation_mode": false,
  "priority_mode": "my_usage_first",
  "auto_pause_gaming": true
}`,
      },
    },
    {
      id: 'security',
      title: 'Security & Privacy',
      description: 'Your data and hardware are protected with enterprise-grade security.',
      icon: <Shield className="h-4 w-4" />,
      benefit: { text: 'Sandboxed execution with encrypted communication', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Security Features',
        content: `✓ Sandboxed execution (no file access)
✓ Encrypted communication (TLS 1.3)
✓ No access to personal data
✓ Malware scanning on all jobs
✓ Automatic security updates
✓ Insurance coverage included`,
      },
    },
    {
      id: 'analytics',
      title: 'Earnings Dashboard',
      description: 'Track your earnings, utilization, and performance in real‑time.',
      icon: <BarChart3 className="h-4 w-4" />,
      benefit: { text: 'Real‑time earnings tracking with historical charts', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Earnings Summary',
        content: `Today:        €42.50
This Week:    €287.30
This Month:   €1,124.80

Utilization:  78%
Avg Rate:     €1.85/hr
Top GPU:      RTX 4090 (€524/mo)`,
      },
    },
    {
      id: 'limits',
      title: 'Usage Limits',
      description: 'Set limits to protect your hardware and control costs.',
      icon: <Sliders className="h-4 w-4" />,
      benefit: { text: 'Temperature monitoring and automatic cooldown periods', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Hardware Protection',
        content: `{
  "max_hours_per_day": 18,
  "temp_limit": 80,
  "power_cap": 350,
  "cooldown_minutes": 15,
  "warranty_mode": true
}`,
      },
    },
    {
      id: 'performance',
      title: 'Performance Optimization',
      description: 'Maximize your earnings with automatic optimization.',
      icon: <Zap className="h-4 w-4" />,
      benefit: { text: 'Automatic model selection and load balancing', tone: 'primary' as const },
      example: {
        kind: 'code' as const,
        title: 'Optimization Stats',
        content: `Idle Detection:     ✓ Active
Auto-Start:         ✓ Enabled
Load Balancing:     2 GPUs
Priority Queue:     High-paying jobs first
Benchmark Score:    9,847 (top 5%)
Earnings Boost:     +23% vs. baseline`,
      },
    },
  ],
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

export const providersUseCasesProps: ProvidersUseCasesTemplateProps = {
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
      image: {
        Component: GamingPcOwner,
        alt: 'illustration of a modern gaming PC setup with RGB-lit tower showing GPU fans through tempered glass panel, dual monitors, and mechanical keyboard with colorful backlighting',
      },
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
      image: {
        Component: HomelabEnthusiast,
        alt: 'illustration of a professional server rack with 19-inch rails, 4U chassis showing multiple GPUs through ventilated panel, blue LED status indicators, and color-coded ethernet cables with cable management',
      },
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
      image: {
        Component: FormerCryptoMiner,
        alt: 'illustration of a repurposed open-air mining frame with aluminum rails, 8 GPUs mounted horizontally with PCIe risers, clean cable management with zip ties, LED strip lighting, and industrial power supply',
      },
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
      image: {
        Component: WorkstationOwner,
        alt: 'illustration of a creative workstation with 34-inch ultrawide curved monitor displaying 3D modeling software, graphics tablet with stylus, and powerful tower with mesh front panel and white LED accents',
      },
    },
  ],
  ctas: {
    primary: { label: 'Start Earning', href: '/signup' },
    secondary: { label: 'Estimate My Payout', href: '#earnings-calculator' },
  },
}

// === ProvidersEarnings Template ===
const fmt = (n: number, opts: Intl.NumberFormatOptions = {}) =>
  new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0, ...opts }).format(n)

const fmtHr = (n: number) => `${fmt(n, { maximumFractionDigits: 2 })}/hr`

export const providersEarningsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Estimate Your Earnings',
  title: 'Calculate Your Potential Earnings',
  description: 'See what you could earn based on GPU model, availability, and utilization.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const providersEarningsProps: ProvidersEarningsTemplateProps = {
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
  disclaimerText:
    'Earnings are estimates based on current market rates and may vary. Actual earnings depend on demand, your pricing, and availability. Figures are estimates.',
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

export const providersMarketplaceProps: ProvidersMarketplaceTemplateProps = {
  featureTiles: [
    {
      icon: TrendingUp,
      title: 'Dynamic Pricing',
      description: 'Set your own rate or use auto-pricing.',
    },
    {
      icon: Users,
      title: 'Growing Demand',
      description: 'Thousands of AI jobs posted monthly.',
    },
    {
      icon: Globe,
      title: 'Global Reach',
      description: 'Your GPUs are discoverable worldwide.',
    },
    {
      icon: Shield,
      title: 'Fair Commission',
      description: 'Keep 85% of every payout.',
    },
  ],
  marketplaceFeaturesTitle: 'Marketplace Features',
  marketplaceFeatures: [
    {
      title: 'Automatic Matching',
      description: 'Jobs match your GPUs based on specs and your pricing.',
    },
    {
      title: 'Rating System',
      description: 'Higher ratings unlock more jobs and better rates.',
    },
    {
      title: 'Guaranteed Payments',
      description: 'Customers pre-pay. Every completed job is paid.',
    },
    {
      title: 'Dispute Resolution',
      description: 'A fair process protects both providers and customers.',
    },
  ],
  commissionStructureTitle: 'Commission Structure',
  standardCommissionLabel: 'Standard Commission',
  standardCommissionValue: '15%',
  standardCommissionDescription: 'Covers marketplace operations, payouts, and support.',
  youKeepLabel: 'You Keep',
  youKeepValue: '85%',
  youKeepDescription: 'No hidden fees or surprise deductions.',
  exampleTitle: 'Example',
  exampleItems: [
    { label: 'Example job', value: '€100.00' },
    { label: 'rbee commission (15%)', value: '−€15.00' },
  ],
  exampleTotalLabel: 'Your earnings',
  exampleTotalValue: '€85.00',
  exampleBadgeText: 'Effective take-home: 85%',
}

// === ProvidersSecurity Template ===
export const providersSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  kicker: 'Security & Trust',
  title: 'Your Security Is Our Priority',
  description: 'Enterprise-grade protections for your hardware, data, and earnings.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const providersSecurityProps: ProvidersSecurityTemplateProps = {
  items: [
    {
      icon: Shield,
      title: 'Sandboxed Execution',
      subtitle: 'Complete isolation',
      body: 'All jobs run in isolated sandboxes with no access to your files, network, or personal data.',
      points: ['No file system access', 'No network access', 'No personal data access', 'Automatic cleanup'],
    },
    {
      icon: Lock,
      title: 'Encrypted Communication',
      subtitle: 'End-to-end encryption',
      body: 'All communication between your GPU and the marketplace is encrypted using industry-standard protocols.',
      points: ['TLS 1.3', 'Secure payment processing', 'Protected earnings data', 'Private job details'],
    },
    {
      icon: Eye,
      title: 'Malware Scanning',
      subtitle: 'Automatic protection',
      body: 'Every job is automatically scanned for malware before execution. Suspicious jobs are blocked.',
      points: ['Real-time detection', 'Automatic blocking', 'Threat intel updates', 'Customer vetting'],
    },
    {
      icon: FileCheck,
      title: 'Hardware Protection',
      subtitle: 'Warranty-safe operation',
      body: 'Temperature monitoring, cooldown periods, and power limits protect your hardware and warranty.',
      points: ['Temperature monitoring', 'Cooldown periods', 'Power limits', 'Health monitoring'],
    },
  ],
  ribbon: { text: 'Plus: €1M insurance coverage is included for all providers—your hardware is protected.' },
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

export const providersTestimonialsProps: ProvidersTestimonialsTemplateProps = {
  sectorFilter: 'provider',
  stats: [
    {
      icon: Users,
      value: '500+',
      label: 'Active Providers',
    },
    {
      icon: Cpu,
      value: '2,000+',
      label: 'GPUs Earning',
    },
    {
      icon: TrendingUp,
      value: '€180K+',
      label: 'Paid to Providers',
    },
    {
      icon: Star,
      value: '4.8/5',
      label: 'Average Rating',
    },
  ],
}

// === ProvidersCTA Template ===
export const providersCTAProps: ProvidersCTATemplateProps = {
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
      icon: Clock,
      value: '< 15 minutes',
      label: 'Setup time',
    },
    {
      icon: Shield,
      value: '15% platform fee',
      label: 'You keep 85%',
    },
    {
      icon: Wallet,
      value: '€25 minimum',
      label: 'Weekly payouts',
    },
  ],
  backgroundImage: {
    src: gpuEarnings,
    alt: 'Cinematic macro shot of three modern NVIDIA RTX GPUs stacked vertically with visible cooling fans and RGB accents, emitting warm amber and orange volumetric light rays from their edges; translucent holographic euro currency symbols (€) and AI task tokens with neural network patterns float upward in a gentle arc representing passive income; dark navy blue to black gradient backdrop with subtle hexagonal mesh pattern; shallow depth of field with bokeh effect; dramatic side lighting creating rim light on GPU edges; photorealistic 3D render style; high contrast with deep shadows; premium tech aesthetic; 8K quality; particles of light dust in the air catching the amber glow; emphasis on AI workload monetization not cryptocurrency mining',
  },
}
