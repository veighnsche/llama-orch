import { Badge } from "@rbee/ui/atoms";
import {
  TemplateContainer,
  type TemplateContainerProps,
} from "@rbee/ui/molecules";
import { faqBeehive, homelabNetwork, pricingHero } from "@rbee/ui/assets";
import { CodeBlock } from "@rbee/ui/molecules/CodeBlock";
import { GPUUtilizationBar } from "@rbee/ui/molecules/GPUUtilizationBar";
import { TerminalWindow } from "@rbee/ui/molecules/TerminalWindow";
import { } from "@rbee/ui/organisms";
import {
  AudienceSelector,
  type AudienceSelectorProps,
  ComparisonTemplate,
  type ComparisonTemplateProps,
  CTATemplate,
  type CTATemplateProps,
  EmailCapture,
  type EmailCaptureProps,
  FAQTemplate,
  type FAQTemplateProps,
  FeaturesTabs,
  type FeaturesTabsProps,
  HomeHero,
  type HomeHeroProps,
  HowItWorks,
  type HowItWorksProps,
  PricingTemplate,
  type PricingTemplateProps,
  ProblemTemplate,
  type ProblemTemplateProps,
  SolutionTemplate,
  type SolutionTemplateProps,
  TechnicalTemplate,
  type TechnicalTemplateProps,
  TestimonialsTemplate,
  type TestimonialsTemplateProps,
  UseCasesTemplate,
  type UseCasesTemplateProps,
  type WhatIsRbeeProps,
  WhatIsRbee,
} from "@rbee/ui/templates";
import { CoreFeaturesTabs } from "@rbee/ui/organisms/CoreFeaturesTabs";
import { ComplianceShield, DevGrid, GpuMarket, RbeeArch } from "@rbee/ui/icons";
import {
  AlertTriangle,
  Anchor,
  ArrowRight,
  BookOpen,
  Building,
  Check,
  Code,
  Code2,
  Cpu,
  DollarSign,
  Gauge,
  Home as HomeIcon,
  Laptop,
  Layers,
  Lock,
  Server,
  Shield,
  Unlock,
  Users,
  Workflow,
  X,
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

// === Problem Template ===
export const problemTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "The hidden risk of AI-assisted development",
  description:
    "You're building complex codebases with AI assistance. What happens when the provider changes the rules?",
  kickerVariant: "destructive",
  bgVariant: "destructive-gradient",
  paddingY: "xl",
  maxWidth: "7xl",
  align: "center",
};

export const problemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      title: "The model changes",
      body: "Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked.",
      icon: <AlertTriangle className="h-6 w-6" />,
      tone: "destructive",
    },
    {
      title: "The price increases",
      body: "$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.",
      icon: <DollarSign className="h-6 w-6" />,
      tone: "primary",
    },
    {
      title: "The provider shuts down",
      body: "APIs get deprecated. Your AI-built code becomes unmaintainable overnight.",
      icon: <Lock className="h-6 w-6" />,
      tone: "destructive",
    },
  ],
};

// === Solution Template ===
export const solutionTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Your hardware. Your models. Your control.",
  description:
    "rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform.",
  bgVariant: "default",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const solutionTemplateProps: SolutionTemplateProps = {
  features: [
    {
      icon: <DollarSign className="h-6 w-6" />,
      title: "Zero ongoing costs",
      body: "Pay only for electricity. No API bills, no per-token surprises.",
    },
    {
      icon: <Shield className="h-6 w-6" />,
      title: "Complete privacy",
      body: "Code and data never leave your network. Audit-ready by design.",
    },
    {
      icon: <Anchor className="h-6 w-6" />,
      title: "Locked to your rules",
      body: "Models update only when you approve. No breaking changes.",
    },
    {
      icon: <Laptop className="h-6 w-6" />,
      title: "Use all your hardware",
      body: "CUDA, Metal, and CPU orchestrated as one pool.",
    },
  ],
  topology: {
    mode: "multi-host",
    hosts: [
      {
        hostLabel: "Gaming PC",
        workers: [
          { id: "w0", label: "GPU 0", kind: "cuda" },
          { id: "w1", label: "GPU 1", kind: "cuda" },
        ],
      },
      {
        hostLabel: "MacBook Pro",
        workers: [{ id: "w2", label: "GPU 0", kind: "metal" }],
      },
      {
        hostLabel: "Workstation",
        workers: [
          { id: "w3", label: "GPU 0", kind: "cuda" },
          { id: "w4", label: "CPU 0", kind: "cpu" },
        ],
      },
    ],
  },
};

// === How It Works ===
export const howItWorksContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "From zero to AI infrastructure in 15 minutes",
  bgVariant: "secondary",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const howItWorksProps: HowItWorksProps = {
  steps: [
    {
      label: "Install rbee",
      block: {
        kind: "terminal",
        title: "terminal",
        lines: (
          <>
            <div>curl -sSL https://rbee.dev/install.sh | sh</div>
            <div className="text-[var(--syntax-comment)]">
              rbee-keeper daemon start
            </div>
          </>
        ),
        copyText:
          "curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start",
      },
    },
    {
      label: "Add your machines",
      block: {
        kind: "terminal",
        title: "terminal",
        lines: (
          <>
            <div>
              rbee-keeper setup add-node --name workstation --ssh-host
              192.168.1.10
            </div>
            <div className="text-[var(--syntax-comment)]">
              rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20
            </div>
          </>
        ),
        copyText:
          "rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20",
      },
    },
    {
      label: "Configure your IDE",
      block: {
        kind: "terminal",
        title: "terminal",
        lines: (
          <>
            <div>
              <span className="text-[var(--syntax-keyword)]">export</span>{" "}
              OPENAI_API_BASE=http://localhost:8080/v1
            </div>
            <div className="text-[var(--syntax-comment)]">
              # OpenAI-compatible endpoint — works with Zed & Cursor
            </div>
          </>
        ),
        copyText: "export OPENAI_API_BASE=http://localhost:8080/v1",
      },
    },
    {
      label: "Build AI agents",
      block: {
        kind: "code",
        title: "TypeScript",
        language: "ts",
        code: `import { invoke } from '@rbee/utils';

const code = await invoke({
  prompt: 'Generate API from schema',
  model: 'llama-3.1-70b'
});`,
      },
    },
  ],
};

// === Features Tabs Section ===
export const featuresTabsProps: FeaturesTabsProps = {
  title: "Core capabilities",
  description: "Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.",
  tabs: [
    {
      value: "api",
      icon: Code,
      label: "OpenAI-Compatible",
      mobileLabel: "API",
      subtitle: "Drop-in API",
      badge: "Drop-in",
      description: "Swap endpoints, keep your code. Works with Zed, Cursor, Continue—any OpenAI client.",
      content: (
        <CodeBlock
          code={`# Before: OpenAI
export OPENAI_API_KEY=sk-...

# After: rbee (same code)
export OPENAI_API_BASE=http://localhost:8080/v1`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: "Drop-in replacement. Point to localhost.",
        variant: "success",
      },
      benefits: [
        { text: "No vendor lock-in" },
        { text: "Use your models + GPUs" },
        { text: "Keep existing tooling" },
      ],
    },
    {
      value: "gpu",
      icon: Cpu,
      label: "Multi-GPU",
      mobileLabel: "GPU",
      subtitle: "Use every GPU",
      badge: "Scale",
      description: "Run across CUDA, Metal, and CPU backends. Use every GPU across your network.",
      content: (
        <div className="space-y-3">
          <GPUUtilizationBar label="RTX 4090 #1" percentage={92} />
          <GPUUtilizationBar label="RTX 4090 #2" percentage={88} />
          <GPUUtilizationBar label="M2 Ultra" percentage={76} />
          <GPUUtilizationBar label="CPU Backend" percentage={34} variant="secondary" />
        </div>
      ),
      highlight: {
        text: "Higher throughput by saturating all devices.",
        variant: "success",
      },
      benefits: [
        { text: "Bigger models fit" },
        { text: "Lower latency under load" },
        { text: "No single-machine bottleneck" },
      ],
    },
    {
      value: "scheduler",
      icon: Gauge,
      label: "Programmable scheduler (Rhai)",
      mobileLabel: "Rhai",
      subtitle: "Route with Rhai",
      badge: "Control",
      description: "Write routing rules. Send 70B to multi-GPU, images to CUDA, everything else to cheapest.",
      content: (
        <CodeBlock
          code={`// Custom routing logic
if task.model.contains("70b") {
  route_to("multi-gpu-cluster")
}
else if task.type == "image" {
  route_to("cuda-only")
}
else {
  route_to("cheapest")
}`}
          language="rust"
          copyable={true}
        />
      ),
      highlight: {
        text: "Optimize for cost, latency, or compliance—your rules.",
        variant: "primary",
      },
      benefits: [
        { text: "Deterministic routing" },
        { text: "Policy & compliance ready" },
        { text: "Easy to evolve" },
      ],
    },
    {
      value: "sse",
      icon: Zap,
      label: "Task-based API with SSE",
      mobileLabel: "SSE",
      subtitle: "Live job stream",
      badge: "Observe",
      description: "See model loading, token generation, and costs stream in as they happen.",
      content: (
        <TerminalWindow
          showChrome={false}
          copyable={true}
          copyText={`→ event: task.created
{ "id": "task_123", "status": "pending" }

→ event: model.loading
{ "progress": 0.45, "eta": "2.1s" }

→ event: token.generated
{ "token": "const", "total": 1 }

→ event: token.generated
{ "token": " api", "total": 2 }`}
        >
          <div className="space-y-2" role="log" aria-live="polite">
            <div role="status">
              <div className="text-muted-foreground">→ event: task.created</div>
              <div className="pl-4">{'{ "id": "task_123", "status": "pending" }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: model.loading</div>
              <div className="pl-4">{'{ "progress": 0.45, "eta": "2.1s" }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: token.generated</div>
              <div className="pl-4">{'{ "token": "const", "total": 1 }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: token.generated</div>
              <div className="pl-4">{'{ "token": " api", "total": 2 }'}</div>
            </div>
          </div>
        </TerminalWindow>
      ),
      highlight: {
        text: "Full visibility for every inference job.",
        variant: "default",
      },
      benefits: [
        { text: "Faster debugging" },
        { text: "UX you can trust" },
        { text: "Accurate cost tracking" },
      ],
    },
  ],
  defaultTab: "api",
};

// === Use Cases Template ===
export const useCasesTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Built for those who value independence",
  description:
    "Run serious AI on your own hardware. Keep costs at zero, keep control at 100%.",
  bgVariant: "secondary",
  paddingY: "2xl",
  maxWidth: "6xl",
  align: "center",
};

export const useCasesTemplateProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <Laptop className="h-6 w-6" />,
      title: "The solo developer",
      scenario:
        "Shipping a SaaS with AI features; wants control without vendor lock-in.",
      solution:
        "Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.",
      outcome: "$0/month AI costs. Full control. No rate limits.",
    },
    {
      icon: <Users className="h-6 w-6" />,
      title: "The small team",
      scenario: "5-person startup burning $500/mo on APIs.",
      solution:
        "Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.",
      outcome: "$6,000+ saved per year. GDPR-friendly by design.",
    },
    {
      icon: <HomeIcon className="h-6 w-6" />,
      title: "The homelab enthusiast",
      scenario: "Four GPUs gathering dust.",
      solution:
        "Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.",
      outcome: "Idle GPUs → productive. Auto-download models, clean shutdowns.",
    },
    {
      icon: <Building className="h-6 w-6" />,
      title: "The enterprise",
      scenario: "50-dev org. Code cannot leave the premises.",
      solution:
        "On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.",
      outcome: "EU-only compliance. Zero external dependencies.",
    },
    {
      icon: <Code className="h-6 w-6" />,
      title: "The AI-dependent coder",
      scenario:
        "Building complex codebases with Claude/GPT-4. Fears provider changes, shutdowns, or price hikes.",
      solution:
        "Build your own AI coders with rbee + llama-orch-utils. OpenAI-compatible API runs on YOUR hardware.",
      outcome:
        "Complete independence. Models never change without permission. $0/month forever.",
    },
    {
      icon: <Workflow className="h-6 w-6" />,
      title: "The agentic AI builder",
      scenario:
        "Needs to build custom AI agents: code generators, doc writers, test creators, code reviewers.",
      solution:
        "Use llama-orch-utils TypeScript library: file ops, LLM invocation, prompt management, response extraction.",
      outcome:
        "Build production AI agents in hours. Full control. No rate limits. Test reproducibility built-in.",
    },
  ],
  columns: 3,
};

// === Comparison Template ===
export const comparisonTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Why Developers Choose rbee",
  description:
    "Local-first AI that's faster, private, and costs $0 on your hardware.",
  bgVariant: "secondary",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const comparisonTemplateProps: ComparisonTemplateProps = {
  columns: [
    { key: "rbee", label: "rbee", accent: true },
    { key: "openai", label: "OpenAI & Anthropic" },
    { key: "ollama", label: "Ollama" },
    { key: "runpod", label: "Runpod & Vast.ai" },
  ],
  rows: [
    {
      feature: "Total Cost",
      values: {
        rbee: "$0 (runs on your hardware)",
        openai: "$20–100/mo per dev",
        ollama: "$0",
        runpod: "$0.50–2/hr",
      },
    },
    {
      feature: "Privacy / Data Residency",
      values: {
        rbee: true,
        openai: false,
        ollama: true,
        runpod: false,
      },
      note: "Complete data control vs. limited",
    },
    {
      feature: "Multi-GPU Utilization",
      values: {
        rbee: true,
        openai: "N/A",
        ollama: "Limited",
        runpod: true,
      },
    },
    {
      feature: "OpenAI-Compatible API",
      values: {
        rbee: true,
        openai: true,
        ollama: "Partial",
        runpod: false,
      },
    },
    {
      feature: "Custom Routing Policies",
      values: {
        rbee: true,
        openai: false,
        ollama: false,
        runpod: false,
      },
    },
    {
      feature: "Rate Limits / Quotas",
      values: {
        rbee: "None",
        openai: "Yes",
        ollama: "None",
        runpod: "Yes",
      },
    },
  ],
  legend: [
    {
      icon: <Check className="h-3.5 w-3.5 text-chart-3" aria-hidden="true" />,
      label: "Available",
    },
    {
      icon: <X className="h-3.5 w-3.5 text-destructive" aria-hidden="true" />,
      label: "Not available",
    },
  ],
  legendNote: '"Partial" = limited coverage',
  footerMessage: "Bring your own GPUs, keep your data in-house.",
  ctas: [
    { label: "See Quickstart", href: "/docs/quickstart" },
    { label: "Architecture", href: "/docs/architecture", variant: "ghost" },
  ],
};

// === Pricing Template ===
export const pricingTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Start Free. Scale When Ready.",
  description:
    "Run rbee free at home. Add collaboration and governance when your team grows.",
  bgVariant: "default",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const pricingTemplateProps: PricingTemplateProps = {
  kickerBadges: [
    {
      icon: <Unlock className="h-3.5 w-3.5" aria-hidden="true" />,
      label: "Open source",
    },
    {
      icon: <Zap className="h-3.5 w-3.5" aria-hidden="true" />,
      label: "OpenAI-compatible",
    },
    {
      icon: <Layers className="h-3.5 w-3.5" aria-hidden="true" />,
      label: "Multi-GPU",
    },
    {
      icon: <Shield className="h-3.5 w-3.5" aria-hidden="true" />,
      label: "No feature gates",
    },
  ],
  tiers: [
    {
      title: "Home/Lab",
      price: "€0",
      period: "forever",
      features: [
        "Unlimited GPUs on your hardware",
        "OpenAI-compatible API",
        "Multi-modal models",
        "Active community support",
        "Open source core",
      ],
      ctaText: "Download rbee",
      ctaHref: "/download",
      ctaVariant: "outline",
      footnote: "Local use. No feature gates.",
      className:
        "col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500",
    },
    {
      title: "Team",
      price: "€99",
      priceYearly: "€990",
      period: "/month",
      features: [
        "Everything in Home/Lab",
        "Web UI for cluster & models",
        "Shared workspaces & quotas",
        "Priority support (business hours)",
        "Rhai policy templates (rate/data)",
      ],
      ctaText: "Start 30-Day Trial",
      ctaHref: "/signup?plan=team",
      highlighted: true,
      badge: "Most Popular",
      footnote: "Cancel anytime during trial.",
      saveBadge: "2 months free",
      className:
        "col-span-12 md:col-span-4 order-first md:order-none motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-100",
    },
    {
      title: "Enterprise",
      price: "Custom",
      features: [
        "Everything in Team",
        "Dedicated, isolated instances",
        "Custom SLAs & onboarding",
        "White-label & SSO options",
        "Enterprise security & support",
      ],
      ctaText: "Contact Sales",
      ctaHref: "/contact?type=enterprise",
      ctaVariant: "outline",
      footnote: "We'll reply within 1 business day.",
      className:
        "col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-200",
    },
  ],
  editorialImage: {
    src: pricingHero,
    alt:
      "Detailed isometric 3D illustration in dark mode showing a progression from left to right: a compact single-GPU homelab server rack (glowing neon teal accents) seamlessly transforming into a large-scale multi-node GPU cluster with interconnected nodes (amber and teal lighting). Clean editorial photography style with dramatic cinematic lighting, sharp focus on hardware details, floating UI panels showing metrics, dark navy background with subtle grid, professional tech marketing aesthetic, 4K quality, Octane render look",
  },
  footer: {
    mainText:
      "Every plan includes the full rbee orchestrator. No feature gates. No artificial limits.",
    subText: "Prices exclude VAT. OSS license applies to Home/Lab.",
  },
};

// === Testimonials Template ===
export const testimonialsTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Trusted by developers who value independence",
  bgVariant: "default",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const testimonialsTemplateProps: TestimonialsTemplateProps = {
  testimonials: [
    {
      avatar: "👨‍💻",
      author: "Alex K.",
      role: "Solo Developer",
      quote:
        "Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.",
    },
    {
      avatar: "👩‍💼",
      author: "Sarah M.",
      role: "CTO",
      quote:
        "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes.",
    },
    {
      avatar: "👨‍🔧",
      author: "Marcus T.",
      role: "DevOps",
      quote:
        "Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.",
    },
  ],
  stats: [
    { value: "1,200+", label: "GitHub stars", valueTone: "foreground" },
    {
      value: "500+",
      label: "Active installations",
      valueTone: "foreground",
    },
    {
      value: "8,000+",
      label: "GPUs orchestrated",
      valueTone: "foreground",
    },
    { value: "€0", label: "Avg. monthly cost", valueTone: "primary" },
  ],
};

// === Technical Template ===
export const technicalTemplateContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: "Built by AI Engineers, for Engineers building with AI",
  description:
    "Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai.",
  bgVariant: "default",
  paddingY: "2xl",
  maxWidth: "7xl",
  align: "center",
};

export const technicalTemplateProps: TechnicalTemplateProps = {
  architectureHighlights: [
    {
      title: "BDD-Driven Development",
      details: ["42/62 scenarios passing (68% complete)", "Live CI coverage"],
    },
    {
      title: "Cascading Shutdown Guarantee",
      details: ["No orphaned processes. Clean VRAM lifecycle."],
    },
    {
      title: "Process Isolation",
      details: ["Worker-level sandboxes. Zero cross-leak."],
    },
    {
      title: "Protocol-Aware Orchestration",
      details: ["SSE, JSON, binary protocols."],
    },
    {
      title: "Smart/Dumb Separation",
      details: ["Central brain, distributed execution."],
    },
  ],
  coverageProgress: {
    label: "BDD Coverage",
    passing: 42,
    total: 62,
  },
  architectureDiagram: {
    component: RbeeArch,
    ariaLabel:
      "rbee architecture diagram showing orchestrator, policy engine, and worker pools",
  },
  techStack: [
    {
      name: "Rust",
      description: "Performance + memory safety.",
      ariaLabel: "Tech: Rust",
    },
    {
      name: "Candle ML",
      description: "Rust-native inference.",
      ariaLabel: "Tech: Candle ML",
    },
    {
      name: "Rhai Scripting",
      description: "Embedded, sandboxed policies.",
      ariaLabel: "Tech: Rhai Scripting",
    },
    {
      name: "SQLite",
      description: "Embedded, zero-ops DB.",
      ariaLabel: "Tech: SQLite",
    },
    {
      name: "Axum + Vue.js",
      description: "Async backend + modern UI.",
      ariaLabel: "Tech: Axum + Vue.js",
    },
  ],
  stackLinks: {
    githubUrl: "https://github.com/veighnsche/llama-orch",
    license: "GPL-3.0-or-later",
    architectureUrl: "/docs/architecture",
  },
};

// === FAQ Template ===
// Note: Self-contained section with internal <section>, no TemplateContainer wrapper
export const faqTemplateProps: FAQTemplateProps = {
  badgeText: "Support • Self-hosted AI",
  categories: [
    "Setup",
    "Models",
    "Performance",
    "Marketplace",
    "Security",
    "Production",
  ],
  faqItems: [
    {
      value: "item-1",
      question: "How is this different from Ollama?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            Ollama excels on a single machine. rbee orchestrates across
            machines and backends (CUDA, Metal, CPU), with an
            OpenAI-compatible, task-based API and SSE streaming—plus a
            programmable scheduler and optional marketplace federation.
          </p>
        </div>
      ),
      category: "Performance",
    },
    {
      value: "item-2",
      question: "Do I need to be a Rust expert?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            No. Use prebuilt binaries via CLI or Web UI. Customize routing
            with simple Rhai scripts or YAML if needed.
          </p>
        </div>
      ),
      category: "Setup",
    },
    {
      value: "item-3",
      question: "What if I don't have GPUs?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            CPU-only works (slower). You can later federate to external GPU
            providers via the marketplace.
          </p>
        </div>
      ),
      category: "Setup",
    },
    {
      value: "item-4",
      question: "Is this production-ready?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            We're in M0 today—great for dev and homelabs. Production SLAs,
            health monitoring, and marketplace land across M1–M3.
          </p>
        </div>
      ),
      category: "Production",
    },
    {
      value: "item-5",
      question: "How do I migrate from OpenAI API?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            Switch one env var:{" "}
            <code>export OPENAI_API_BASE=http://localhost:8080/v1</code>
          </p>
        </div>
      ),
      category: "Setup",
    },
    {
      value: "item-6",
      question: "What models are supported?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            Any GGUF from Hugging Face (Llama, Mistral, Qwen, DeepSeek). Image
            gen and TTS arrive in M2.
          </p>
        </div>
      ),
      category: "Models",
    },
    {
      value: "item-7",
      question: "Can I sell GPU time?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            Yes—via the marketplace in M3: register your node and earn from
            excess capacity.
          </p>
        </div>
      ),
      category: "Marketplace",
    },
    {
      value: "item-8",
      question: "What about security?",
      answer: (
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
          <p>
            Runs entirely on your network. Rhai scripts are sandboxed (time &
            memory limits). Platform mode uses immutable schedulers for
            multi-tenant isolation.
          </p>
        </div>
      ),
      category: "Security",
    },
  ],
  supportCard: {
    image: faqBeehive,
    imageAlt:
      "Isometric illustration of a vibrant community hub: hexagonal beehive structure with worker bees collaborating around glowing question mark icons, speech bubbles floating between honeycomb cells containing miniature server racks, warm amber and honey-gold palette with soft cyan accents, friendly bees wearing tiny headsets offering support, knowledge base documents scattered on wooden surface, gentle directional lighting creating welcoming atmosphere, detailed technical diagrams visible through translucent honeycomb walls, community-driven support concept, approachable and helpful mood",
    title: "Still stuck?",
    links: [
      {
        label: "Join Discussions",
        href: "https://github.com/veighnsche/llama-orch/discussions",
      },
      { label: "Read Setup Guide", href: "/docs/setup" },
      { label: "Email support", href: "mailto:support@rbee.dev" },
    ],
    cta: {
      label: "Open Discussions",
      href: "https://github.com/veighnsche/llama-orch/discussions",
    },
  },
  jsonLdEnabled: true,
};

// === CTA Template ===
// Note: Self-contained section with internal <section>, no TemplateContainer wrapper
export const ctaTemplateProps: CTATemplateProps = {
  title: "Stop depending on AI providers. Start building today.",
  subtitle:
    "Join 500+ developers who've taken control of their AI infrastructure.",
  primary: {
    label: "Get started free",
    href: "/getting-started",
    iconRight: ArrowRight,
  },
  secondary: {
    label: "View documentation",
    href: "/docs",
    iconLeft: BookOpen,
    variant: "outline",
  },
  note: "100% open source. No credit card required. Install in 15 minutes.",
  emphasis: "gradient",
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
      <TemplateContainer {...problemTemplateContainerProps}>
        <ProblemTemplate {...problemTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...solutionTemplateContainerProps}>
        <SolutionTemplate {...solutionTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...howItWorksContainerProps}>
        <HowItWorks {...howItWorksProps} />
      </TemplateContainer>
      <FeaturesTabs {...featuresTabsProps} />
      <TemplateContainer {...useCasesTemplateContainerProps}>
        <UseCasesTemplate {...useCasesTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...comparisonTemplateContainerProps}>
        <ComparisonTemplate {...comparisonTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...pricingTemplateContainerProps}>
        <PricingTemplate {...pricingTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...testimonialsTemplateContainerProps}>
        <TestimonialsTemplate {...testimonialsTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...technicalTemplateContainerProps}>
        <TechnicalTemplate {...technicalTemplateProps} />
      </TemplateContainer>
      <FAQTemplate {...faqTemplateProps} />
      <CTATemplate {...ctaTemplateProps} />
    </main>
  );
}
