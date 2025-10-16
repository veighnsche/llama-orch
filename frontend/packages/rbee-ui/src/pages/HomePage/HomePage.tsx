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
  PricingSection,
  TechnicalSection,
  TestimonialsSection,
} from "@rbee/ui/organisms";
import {
  AudienceSelector,
  type AudienceSelectorProps,
  EmailCapture,
  type EmailCaptureProps,
  FeaturesTabs,
  type FeaturesTabsProps,
  HomeHero,
  type HomeHeroProps,
  HowItWorks,
  type HowItWorksProps,
  ProblemTemplate,
  type ProblemTemplateProps,
  SolutionTemplate,
  type SolutionTemplateProps,
  UseCasesTemplate,
  type UseCasesTemplateProps,
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
      icon: Laptop,
      title: "The solo developer",
      scenario:
        "Shipping a SaaS with AI features; wants control without vendor lock-in.",
      solution:
        "Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.",
      outcome: "$0/month AI costs. Full control. No rate limits.",
    },
    {
      icon: Users,
      title: "The small team",
      scenario: "5-person startup burning $500/mo on APIs.",
      solution:
        "Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.",
      outcome: "$6,000+ saved per year. GDPR-friendly by design.",
    },
    {
      icon: HomeIcon,
      title: "The homelab enthusiast",
      scenario: "Four GPUs gathering dust.",
      solution:
        "Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.",
      outcome: "Idle GPUs → productive. Auto-download models, clean shutdowns.",
    },
    {
      icon: Building,
      title: "The enterprise",
      scenario: "50-dev org. Code cannot leave the premises.",
      solution:
        "On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.",
      outcome: "EU-only compliance. Zero external dependencies.",
    },
    {
      icon: Code,
      title: "The AI-dependent coder",
      scenario:
        "Building complex codebases with Claude/GPT-4. Fears provider changes, shutdowns, or price hikes.",
      solution:
        "Build your own AI coders with rbee + llama-orch-utils. OpenAI-compatible API runs on YOUR hardware.",
      outcome:
        "Complete independence. Models never change without permission. $0/month forever.",
    },
    {
      icon: Workflow,
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
    </main>
  );
}
