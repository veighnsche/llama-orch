import { faqBeehive } from '@rbee/ui/assets'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import {
  AudienceSelector,
  ComparisonSection,
  CTASection,
  EmailCapture,
  FAQSection,
  HomeHero,
  type HomeHeroProps,
  HomeSolutionSection,
  HowItWorksSection,
  PricingSection,
  ProblemSection,
  TechnicalSection,
  TestimonialsSection,
  UseCasesSection,
  WhatIsRbee,
} from '@rbee/ui/organisms'
import { CoreFeaturesTabs } from '@rbee/ui/organisms/CoreFeaturesTabs'
import {
  AlertTriangle,
  Anchor,
  ArrowRight,
  BookOpen,
  Building,
  Code,
  Cpu,
  DollarSign,
  Gauge,
  Home as HomeIcon,
  Laptop,
  Lock,
  Shield,
  Users,
  Workflow,
  Zap,
} from 'lucide-react'

export const homeHeroProps: HomeHeroProps = {
  badgeText: '100% Open Source • GPL-3.0-or-later',
  headlinePrefix: 'AI Infrastructure.',
  headlineHighlight: 'On Your Terms.',
  subcopy: 'Run LLMs on your hardware—across any GPUs and machines. Build with AI, keep control, and avoid vendor lock-in.',
  bullets: [
    { title: 'Your GPUs, your network', variant: 'check', color: 'chart-3' },
    { title: 'Zero API fees', variant: 'check', color: 'chart-3' },
    { title: 'Drop-in OpenAI API', variant: 'check', color: 'chart-3' },
  ],
  primaryCTA: {
    label: 'Get Started Free',
    href: '/getting-started',
    showIcon: true,
    dataUmamiEvent: 'cta:get-started',
  },
  secondaryCTA: {
    label: 'View Docs',
    href: '/docs',
    variant: 'outline',
  },
  trustBadges: [
    {
      type: 'github',
      label: 'Star on GitHub',
      href: 'https://github.com/veighnsche/llama-orch',
    },
    {
      type: 'api',
      label: 'OpenAI-Compatible',
    },
    {
      type: 'cost',
      label: '$0 • No Cloud Required',
    },
  ],
  terminalTitle: 'rbee-keeper',
  terminalCommand: 'rbee-keeper infer --model llama-3.1-70b',
  terminalOutput: {
    loading: 'Loading model across 3 GPUs...',
    ready: 'Model ready (2.3s)',
    prompt: 'Generate REST API',
    generating: 'Generating code...',
  },
  gpuPoolLabel: 'GPU Pool (5 nodes):',
  gpuProgress: [
    { label: 'Gaming PC 1', percentage: 91 },
    { label: 'Gaming PC 2', percentage: 88 },
    { label: 'Gaming PC 3', percentage: 76 },
    { label: 'Workstation', percentage: 85 },
  ],
  costLabel: 'Local Inference',
  costValue: '$0.00',
}