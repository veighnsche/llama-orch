import { Badge } from "@rbee/ui/atoms";
import {
  TemplateContainer,
  type TemplateContainerProps,
} from "@rbee/ui/molecules";
import { faqBeehive, homelabNetwork } from "@rbee/ui/assets";
import { CodeBlock } from "@rbee/ui/molecules/CodeBlock";
import { GPUUtilizationBar } from "@rbee/ui/molecules/GPUUtilizationBar";
import { TerminalWindow } from "@rbee/ui/molecules/TerminalWindow";
import {
  ComparisonSection,
  CTASection,
  FAQSection,
  HomeSolutionSection,
  HowItWorksSection,
  PricingSection,
  ProblemSection,
  TechnicalSection,
  TestimonialsSection,
  UseCasesSection,
} from "@rbee/ui/organisms";
import {
  AudienceSelector,
  type AudienceSelectorProps,
  EmailCapture,
  type EmailCaptureProps,
  HomeHero,
  type HomeHeroProps,
  type WhatIsRbeeProps,
  WhatIsRbee,
} from "@rbee/ui/templates";
import { CoreFeaturesTabs } from "@rbee/ui/organisms/CoreFeaturesTabs";
import { ComplianceShield, DevGrid, GpuMarket } from "@rbee/ui/icons";
import {
  AlertTriangle,
  Anchor,
  ArrowRight,
  BookOpen,
  Building,
  Code,
  Code2,
  Cpu,
  DollarSign,
  Gauge,
  Home as HomeIcon,
  Laptop,
  Lock,
  Server,
  Shield,
  Users,
  Workflow,
  Zap,
} from "lucide-react";

export const homeHeroProps: HomeHeroProps = {
  badgeText: "100% Open Source • GPL-3.0-or-later",
  headlinePrefix: "AI Infrastructure.",
  headlineHighlight: "On Your Terms.",
  subcopy:
    "Run LLMs on your hardware—across any GPUs and machines. Build with AI, keep control, and avoid vendor lock-in.",
  bullets: [
    { title: "Your GPUs, your network", variant: "check", color: "chart-3" },
    { title: "Zero API fees", variant: "check", color: "chart-3" },
    { title: "Drop-in OpenAI API", variant: "check", color: "chart-3" },
  ],
  primaryCTA: {
    label: "Get Started Free",
    href: "/getting-started",
    showIcon: true,
    dataUmamiEvent: "cta:get-started",
  },
  secondaryCTA: {
    label: "View Docs",
    href: "/docs",
    variant: "outline",
  },
  trustBadges: [
    {
      type: "github",
      label: "Star on GitHub",
      href: "https://github.com/veighnsche/llama-orch",
    },
    {
      type: "api",
      label: "OpenAI-Compatible",
    },
    {
      type: "cost",
      label: "$0 • No Cloud Required",
    },
  ],
  terminalTitle: "rbee-keeper",
  terminalCommand: "rbee-keeper infer --model llama-3.1-70b",
  terminalOutput: {
    loading: "Loading model across 3 GPUs...",
    ready: "Model ready (2.3s)",
    prompt: "Generate REST API",
    generating: "Generating code...",
  },
  gpuPoolLabel: "GPU Pool (5 nodes):",
  gpuProgress: [
    { label: "Gaming PC 1", percentage: 91 },
    { label: "Gaming PC 2", percentage: 88 },
    { label: "Gaming PC 3", percentage: 76 },
    { label: "Workstation", percentage: 85 },
  ],
  costLabel: "Local Inference",
  costValue: "$0.00",
};

export const whatIsRbeeContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  eyebrow: (
    <Badge variant="secondary" className="uppercase tracking-wide">
      Open-source • Self-hosted
    </Badge>
  ),
  title: "What is rbee?",
  bgVariant: "secondary",
  maxWidth: "5xl",
  paddingY: "xl",
  align: "center",
};

export const whatIsRbeeProps: WhatIsRbeeProps = {
  headlinePrefix: ": your private AI infrastructure",
  headlineSuffix: "",
  pronunciationText: 'pronounced "are-bee"',
  pronunciationTooltip: 'Pronounced like "R.B."',
  description:
    "is an open-source AI orchestration platform that unifies every computer in your home or office into a single, OpenAI-compatible AI cluster—private, controllable, and yours forever.",
  features: [
    {
      icon: Zap,
      title: "Independence",
      description:
        "Build on your hardware. No surprise model or pricing changes.",
    },
    {
      icon: Shield,
      title: "Privacy",
      description: "Code and data never leave your network.",
    },
    {
      icon: Cpu,
      title: "All GPUs together",
      description: "CUDA, Metal, and CPU—scheduled as one.",
    },
  ],
  stats: [
    { value: "$0", label: "No API fees" },
    { value: "100%", label: "Private" },
    { value: "All", label: "CUDA · Metal · CPU" },
  ],
  primaryCTA: {
    label: "Get Started Free",
    href: "#get-started",
  },
  secondaryCTA: {
    label: "See Architecture",
    href: "/technical-deep-dive",
    variant: "ghost",
    showIcon: true,
  },
  closingCopyLine1: "OpenAI-compatible API. Zed/Cursor-ready.",
  closingCopyLine2: "Your models, your rules.",
  visualImage: homelabNetwork,
  visualImageAlt:
    "Distributed homelab AI network diagram showing a central orchestrator mini PC coordinating multiple worker nodes (gaming PCs, workstation, Mac Studio) connected via ethernet cables in a star topology. Each node displays GPU utilization and the network shows zero-cost local inference.",
  visualBadgeText: "Local Network",
};

export const audienceSelectorContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  eyebrow: "Choose your path",
  title: "Where should you start?",
  description:
    "rbee adapts to how you work—build on your own GPUs, monetize idle capacity, or deploy compliant AI at scale.",
  bgVariant: "subtle",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const audienceSelectorProps: AudienceSelectorProps = {
  cards: [
    {
      icon: Code2,
      category: "For Developers",
      title: "Build on Your Hardware",
      description:
        "Power Zed, Cursor, and your own agents on YOUR GPUs. OpenAI-compatible—drop-in, zero API fees.",
      features: [
        "Zero API costs, unlimited usage",
        "Your code stays on your network",
        "Agentic API + TypeScript utils",
      ],
      href: "/developers",
      ctaText: "Explore Developer Path",
      color: "chart-2",
      imageSlot: <DevGrid size={56} aria-hidden />,
      badgeSlot: (
        <Badge
          variant="outline"
          className="border-chart-2/30 bg-chart-2/5 text-chart-2"
        >
          Homelab-ready
        </Badge>
      ),
      decisionLabel: "Code with AI locally",
    },
    {
      icon: Server,
      category: "For GPU Owners",
      title: "Monetize Your Hardware",
      description:
        "Join the rbee marketplace and earn from gaming rigs to server farms—set price, stay in control.",
      features: [
        "Set pricing & availability",
        "Audit trails and payouts",
        "Passive income from idle GPUs",
      ],
      href: "/gpu-providers",
      ctaText: "Become a Provider",
      color: "chart-3",
      imageSlot: <GpuMarket size={56} aria-hidden />,
      decisionLabel: "Earn from idle GPUs",
    },
    {
      icon: Shield,
      category: "For Enterprise",
      title: "Compliance & Security",
      description:
        "EU-native compliance, audit trails, and zero-trust architecture—from day one.",
      features: [
        "GDPR with 7-year retention",
        "SOC2 & ISO 27001 aligned",
        "Private cloud or on-prem",
      ],
      href: "/enterprise",
      ctaText: "Enterprise Solutions",
      color: "primary",
      imageSlot: <ComplianceShield size={56} aria-hidden />,
      decisionLabel: "Deploy with compliance",
    },
  ],
  helperLinks: [
    { label: "Not sure? Compare paths", href: "#compare" },
    { label: "Talk to us", href: "#contact" },
  ],
};

// === Email Capture Section ===
export const emailCaptureProps: EmailCaptureProps = {
  badge: {
    text: "In Development · M0 · 68%",
    showPulse: true,
  },
  headline: "Get Updates. Own Your AI.",
  subheadline:
    "Join the rbee waitlist to get early access, build notes, and launch perks for running AI on your own hardware.",
  emailInput: {
    placeholder: "you@company.com",
    label: "Email address",
  },
  submitButton: {
    label: "Join Waitlist",
  },
  trustMessage: "No spam. Unsubscribe anytime.",
  successMessage: "Thanks! You're on the list — we'll keep you posted.",
  communityFooter: {
    text: "Follow progress & contribute on GitHub",
    linkText: "View Repository",
    linkHref: "https://github.com/veighnsche/llama-orch",
    subtext: "Weekly dev notes. Roadmap issues tagged M0–M2.",
  },
  showBeeGlyphs: true,
  showIllustration: true,
};

export default function HomePage() {
  return (
    <main>
      <HomeHero {...homeHeroProps} />
      <TemplateContainer {...whatIsRbeeContainerProps}>
        <WhatIsRbee {...whatIsRbeeProps} />
      </TemplateContainer>
      <TemplateContainer {...audienceSelectorContainerProps}>
        <AudienceSelector {...audienceSelectorProps} />
      </TemplateContainer>
      <EmailCapture {...emailCaptureProps} />
    </main>
  );
}
