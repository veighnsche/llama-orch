import { homelabHardwareMontage } from '@rbee/ui/assets'
import { Alert, AlertDescription, Card, CardContent, CardHeader, CardTitle, GitHubIcon } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import type {
  CodeExamplesTemplateProps,
  CTATemplateProps,
  DevelopersHeroProps,
  EmailCaptureProps,
  FeaturesTabsProps,
  HowItWorksProps,
  PricingTemplateProps,
  ProblemTemplateProps,
  SolutionTemplateProps,
  TestimonialsTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  ArrowRight,
  Code,
  Cpu,
  DollarSign,
  FileText,
  FlaskConical,
  Gauge,
  GitPullRequest,
  Lock,
  Wrench,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Developers Hero ===

/**
 * Developers Hero - Above-the-fold hero section for developers
 */
export const developersHeroProps: DevelopersHeroProps = {
  badge: {
    text: 'For developers who build with AI',
    showPulse: true,
  },
  headlineFirstLine: 'Build with AI.',
  headlineSecondLine: 'Own your infrastructure.',
  subheadline: (
    <>
      Stop depending on external AI. <strong className="font-semibold text-foreground">rbee</strong> (pronounced
      &quot;are-bee&quot;) gives you an OpenAI-compatible API that runs on{' '}
      <strong className="font-semibold text-foreground">ALL your home network hardware</strong>
      —GPUs, Macs, workstations—with <strong className="font-semibold text-foreground">zero ongoing costs</strong>.
    </>
  ),
  primaryCta: {
    label: 'Get started free',
    href: '#get-started',
  },
  secondaryCta: {
    label: 'View on GitHub',
    href: 'https://github.com/veighnsche/llama-orch',
  },
  tertiaryLink: {
    label: 'How it works',
    href: '#how-it-works',
  },
  trustBadges: ['Open source (GPL-3.0)', 'OpenAI-compatible API', 'Works with Zed & Cursor', 'No cloud required'],
  terminal: {
    title: 'terminal',
    command: 'rbee-keeper infer --model llama-3.1-70b --prompt "Generate API"',
    output: (
      <div className="space-y-1 text-foreground pt-2">
        <div>
          <span className="text-chart-2">export</span> <span className="text-primary">async</span>{' '}
          <span className="text-chart-4">function</span> <span className="text-chart-3">getUsers</span>() {'{'}
        </div>
        <div className="pl-4">
          <span className="text-chart-2">const</span> response = <span className="text-chart-2">await</span>{' '}
          <span className="text-chart-3">fetch</span>(<span className="text-primary">&apos;/api/users&apos;</span>)
        </div>
        <div className="pl-4">
          <span className="text-chart-2">return</span> response.
          <span className="text-chart-3">json</span>()
        </div>
        <div>{'}'}</div>
      </div>
    ),
    stats: {
      gpu1: '87%',
      gpu2: '92%',
      cost: '$0.00',
    },
  },
  hardwareImage: {
    src: homelabHardwareMontage,
    alt: 'Professional product photography of a modern homelab setup on a dark wooden desk with warm ambient lighting: foreground shows a matte black GPU tower PC with subtle RGB accents and visible PCIe slots, mid-ground features a silver M-series MacBook Pro with glowing Apple logo, background includes a compact mini-ITX workstation with exposed heatsinks and a consumer-grade WiFi router with antenna array. Shallow depth of field creates bokeh effect on background elements. Organized cable management with braided black cables. Dark navy gradient backdrop (hex #0f172a to #1e293b). Matte finishes throughout, no glossy surfaces. Studio lighting creates soft highlights on metal chassis. Conveys distributed computing across ALL your home network hardware working in harmony.',
  },
  imageOverlayBadges: ['Zed & Cursor: drop-in via OpenAI API', 'Zero ongoing costs'],
}

// === Email Capture ===

/**
 * Email Capture - Developer-focused messaging
 */
export const developersEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'For Developers',
    showPulse: false,
  },
  headline: 'Build AI tools without vendor lock-in',
  subheadline: "Join developers who've taken control of their AI infrastructure.",
  emailInput: {
    placeholder: 'your@email.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Get Started',
  },
  trustMessage: '100% open source. No credit card required.',
  successMessage: 'Thanks! Check your inbox for next steps.',
  communityFooter: {
    text: 'Join the community',
    linkText: 'GitHub',
    linkHref: 'https://github.com/veighnsche/llama-orch',
    subtext: 'Star us on GitHub or join our Discord',
  },
}

/**
 * Email capture container - Background with bee glyph decorations
 */
export const developersEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}

// === Problem Template ===

/**
 * Problem Template container - Layout configuration
 */
export const problemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Hidden Risk of AI-Assisted Development',
  description:
    "You're building complex codebases with AI assistance. But what happens when your provider changes the rules?",
  bgVariant: 'destructive-gradient',
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
  ctaBanner: {
    copy: 'Stop depending on external AI providers. Build your own AI infrastructure with rbee—your hardware, your models, your control.',
    primary: { label: 'Get Started Free', href: '/getting-started' },
    secondary: { label: 'View Documentation', href: '/docs' },
  },
}

/**
 * Problem Template - The Hidden Risk of AI-Assisted Development
 */
export const problemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      title: 'The Model Changes',
      body: 'Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked.',
      icon: <AlertTriangle className="h-6 w-6" />,
      tone: 'destructive',
      tag: 'High risk',
    },
    {
      title: 'The Price Increases',
      body: '$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control.',
      icon: <DollarSign className="h-6 w-6" />,
      tone: 'primary',
      tag: 'Cost increase: 10x',
    },
    {
      title: 'The Provider Shuts Down',
      body: 'API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight.',
      icon: <Lock className="h-6 w-6" />,
      tone: 'destructive',
      tag: 'Critical failure',
    },
  ],
}

// === Solution Template ===

/**
 * Solution Template container - Layout configuration
 */
export const solutionTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Your Hardware. Your Models. Your Control.',
  description:
    'rbee orchestrates AI inference across every device in your home network, turning idle hardware into a private, OpenAI-compatible AI platform.',
  bgVariant: 'background',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctas: {
    primary: {
      label: 'Get Started',
      href: '/getting-started',
    },
    secondary: {
      label: 'View Documentation',
      href: '/docs',
    },
  },
}

/**
 * Solution Template - Your Hardware. Your Models. Your Control.
 */
export const solutionTemplateProps: SolutionTemplateProps = {
  features: [
    {
      icon: <DollarSign className="h-8 w-8" aria-hidden="true" />,
      title: 'Zero Ongoing Costs',
      body: 'Pay only for electricity. No subscriptions or per-token fees.',
    },
    {
      icon: <Lock className="h-8 w-8" aria-hidden="true" />,
      title: 'Complete Privacy',
      body: 'Code never leaves your network. GDPR-friendly by default.',
    },
    {
      icon: <Zap className="h-8 w-8" aria-hidden="true" />,
      title: 'You Decide When to Update',
      body: 'Models change only when you choose—no surprise breakages.',
    },
    {
      icon: <Cpu className="h-8 w-8" aria-hidden="true" />,
      title: 'Use All Your Hardware',
      body: 'Orchestrate CUDA, Metal, and CPU. Every chip contributes.',
    },
  ],
  steps: [
    {
      title: 'Install rbee',
      body: 'Run one command on Windows, macOS, or Linux.',
    },
    {
      title: 'Add Your Hardware',
      body: 'rbee auto-detects GPUs and CPUs across your network.',
    },
    {
      title: 'Download Models',
      body: 'Pull models from Hugging Face or load local GGUF files.',
    },
    {
      title: 'Start Building',
      body: 'OpenAI-compatible API. Drop-in replacement for your existing code.',
    },
  ],
  aside: (
    <Card className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
      <CardHeader>
        <CardTitle className="text-sm">OpenAI-Compatible API</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <CodeBlock
          language="typescript"
          code={`import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
});

const response = await client.chat
  .completions.create({
    model: 'llama-3.1-70b',
    messages: [{ role: 'user',
      content: 'Hello!' }]
  })`}
          copyable={true}
        />
        <Alert variant="info">
          <AlertDescription className="text-xs">Works with Cursor, Zed, Continue, and any OpenAI SDK</AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  ),
}

// === How It Works Template ===

/**
 * How It Works container - Layout configuration
 */
export const howItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'From zero to AI infrastructure in 15 minutes',
  bgVariant: 'secondary',
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * How It Works - From zero to AI infrastructure in 15 minutes
 */
export const howItWorksProps: HowItWorksProps = {
  steps: [
    {
      label: 'Install rbee',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>curl -sSL https://rbee.dev/install.sh | sh</div>
            <div className="text-slate-400">rbee-keeper daemon start</div>
          </>
        ),
        copyText: 'curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start',
      },
    },
    {
      label: 'Add your machines',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
            <div className="text-slate-400">rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20</div>
          </>
        ),
        copyText:
          'rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20',
      },
    },
    {
      label: 'Configure your IDE',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>
              <span className="text-blue-400">export</span> OPENAI_API_BASE=http://localhost:8080/v1
            </div>
            <div className="text-slate-400"># Now Zed, Cursor, or any OpenAI-compatible tool works!</div>
          </>
        ),
        copyText: 'export OPENAI_API_BASE=http://localhost:8080/v1',
      },
    },
    {
      label: 'Build AI agents',
      block: {
        kind: 'code',
        title: 'TypeScript',
        language: 'ts',
        code: `import { invoke } from '@rbee/utils';

const code = await invoke({
  prompt: 'Generate API from schema',
  model: 'llama-3.1-70b'
});`,
      },
    },
  ],
}

// === Core Features Tabs ===

/**
 * Core Features Tabs - Four core capabilities (API, GPU, Scheduler, SSE)
 */
export const coreFeatureTabsProps: FeaturesTabsProps = {
  title: 'Core capabilities',
  description: 'Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.',
  tabs: [
    {
      value: 'api',
      icon: <Code className="size-6" />,
      label: 'OpenAI-Compatible',
      mobileLabel: 'API',
      subtitle: 'Drop-in API',
      badge: 'Drop-in',
      description: 'Swap endpoints, keep your code. Works with Zed, Cursor, Continue—any OpenAI client.',
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
        text: 'Drop-in replacement. Point to localhost.',
        variant: 'success',
      },
      benefits: [{ text: 'No vendor lock-in' }, { text: 'Use your models + GPUs' }, { text: 'Keep existing tooling' }],
    },
    {
      value: 'gpu',
      icon: <Cpu className="size-6" />,
      label: 'Multi-GPU',
      mobileLabel: 'GPU',
      subtitle: 'Use every GPU',
      badge: 'Scale',
      description: 'Run across CUDA, Metal, and CPU backends. Use every GPU across your network.',
      content: (
        <div className="space-y-3">
          <GPUUtilizationBar label="RTX 4090 #1" percentage={92} />
          <GPUUtilizationBar label="RTX 4090 #2" percentage={88} />
          <GPUUtilizationBar label="M2 Ultra" percentage={76} />
          <GPUUtilizationBar label="CPU Backend" percentage={34} variant="secondary" />
        </div>
      ),
      highlight: {
        text: 'Higher throughput by saturating all devices.',
        variant: 'success',
      },
      benefits: [
        { text: 'Bigger models fit' },
        { text: 'Lower latency under load' },
        { text: 'No single-machine bottleneck' },
      ],
    },
    {
      value: 'scheduler',
      icon: <Gauge className="size-6" />,
      label: 'Programmable scheduler (Rhai)',
      mobileLabel: 'Rhai',
      subtitle: 'Route with Rhai',
      badge: 'Control',
      description: 'Write routing rules. Send 70B to multi-GPU, images to CUDA, everything else to cheapest.',
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
        text: 'Optimize for cost, latency, or compliance—your rules.',
        variant: 'primary',
      },
      benefits: [{ text: 'Deterministic routing' }, { text: 'Policy & compliance ready' }, { text: 'Easy to evolve' }],
    },
    {
      value: 'sse',
      icon: <Zap className="size-6" />,
      label: 'Task-based API with SSE',
      mobileLabel: 'SSE',
      subtitle: 'Live job stream',
      badge: 'Observe',
      description: 'See model loading, token generation, and costs stream in as they happen.',
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
        text: 'Full visibility for every inference job.',
        variant: 'default',
      },
      benefits: [{ text: 'Faster debugging' }, { text: 'UX you can trust' }, { text: 'Accurate cost tracking' }],
    },
  ],
  defaultTab: 'api',
};

/**
 * Core features tabs container
 */
export const coreFeatureTabsContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: "2xl",
  maxWidth: "7xl",
};

// === Use Cases Template ===

/**
 * Use Cases container - Layout configuration
 */
export const useCasesTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Built for developers who value independence',
  bgVariant: 'secondary',
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Use Cases Template - Built for developers who value independence
 */
export const useCasesTemplateProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <Code className="h-6 w-6" />,
      title: 'Build your own AI coder',
      scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
      solution: 'Run rbee on a gaming PC + old workstation. Llama-3.1-70B for code; Stable Diffusion for assets.',
      outcome: '$0/month AI costs. Full control. No rate limits.',
      tags: ['OpenAI-compatible', 'Local models'],
    },
    {
      icon: <FileText className="h-6 w-6" />,
      title: 'Documentation generators',
      scenario: 'Need comprehensive docs from codebase; API costs are prohibitive.',
      solution: 'Process entire repos locally with rbee. Generate markdown with examples.',
      outcome: 'Process entire repos. Zero API costs. Private by default.',
      tags: ['Markdown', 'Privacy'],
    },
    {
      icon: <FlaskConical className="h-6 w-6" />,
      title: 'Test generators',
      scenario: 'Writing tests is time-consuming; need AI to generate comprehensive suites.',
      solution: 'Use rbee + llama-orch-utils to generate Jest/Vitest tests from specs.',
      outcome: '10× faster coverage. No external dependencies.',
      tags: ['Jest', 'Vitest'],
    },
    {
      icon: <GitPullRequest className="h-6 w-6" />,
      title: 'Code review agents',
      scenario: 'Small team needs automated code review but cannot afford enterprise tools.',
      solution: 'Build custom review agent with rbee. Analyze PRs for issues, security, performance.',
      outcome: 'Automated reviews. Zero ongoing costs. Custom rules.',
      tags: ['GitHub', 'GitLab'],
    },
    {
      icon: <Wrench className="h-6 w-6" />,
      title: 'Refactoring agents',
      scenario: 'Legacy codebase needs modernization; manual refactoring would take months.',
      solution: 'Use rbee to refactor code to modern patterns. TypeScript, async/await, etc.',
      outcome: 'Months of work → days. You approve every change.',
      tags: ['TypeScript', 'Modernization'],
    },
  ],
}

/**
 * Developers Code Examples - Build AI agents with llama-orch-utils
 */
export const codeExamplesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Build AI agents with llama-orch-utils',
  description: 'TypeScript utilities for LLM pipelines and agentic workflows.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

export const codeExamplesProps: CodeExamplesTemplateProps = {
  footerNote: 'Works with any OpenAI-compatible client.',
  items: [
    {
      id: 'simple',
      title: 'Simple code generation',
      summary: 'Invoke to generate a TypeScript validator.',
      language: 'TypeScript',
      code: `import { invoke } from '@llama-orch/utils';

const response = await invoke({
  prompt: 'Generate a TypeScript function that validates email addresses',
  model: 'llama-3.1-70b',
  maxTokens: 500
});

console.log(response.text);`,
    },
    {
      id: 'files',
      title: 'File operations',
      summary: 'Read schema → generate API → write file.',
      language: 'TypeScript',
      code: `import { FileReader, FileWriter, invoke } from '@llama-orch/utils';

// Read schema
const schema = await FileReader.read('schema.sql');

// Generate API
const code = await invoke({
  prompt: \`Generate TypeScript CRUD API for:\\n\${schema}\`,
  model: 'llama-3.1-70b'
});

// Write result
await FileWriter.write('src/api.ts', code.text);`,
    },
    {
      id: 'agent',
      title: 'Multi-step agent',
      summary: 'Threaded review + suggestion extraction.',
      language: 'TypeScript',
      code: `import { Thread, invoke, extractCode } from '@llama-orch/utils';

// Build conversation thread
const thread = Thread.create()
  .addSystem('You are a code review expert')
  .addUser('Review this code for security issues')
  .addUser(await FileReader.read('src/auth.ts'));

// Get review
const review = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-70b'
});

// Extract suggestions
const suggestions = extractCode(review.text, 'typescript');
await FileWriter.write('review.md', review.text);`,
    },
  ],
}

// === PricingTemplate ===

/**
 * Pricing template container - wraps the pricing tiers section for developers page
 */
export const developersPricingTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Start Free. Scale When Ready.',
  description: 'Run rbee free at home. Add collaboration and governance when your team grows.',
  bgVariant: 'default',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Pricing template data - three pricing tiers for developers page (no kicker, no editorial image)
 */
export const developersPricingTemplateProps: PricingTemplateProps = {
  tiers: [
    {
      title: 'Home/Lab',
      price: '€0',
      period: 'forever',
      features: [
        'Unlimited GPUs on your hardware',
        'OpenAI-compatible API',
        'Multi-modal models',
        'Active community support',
        'Open source core',
      ],
      ctaText: 'Download rbee',
      ctaHref: '/download',
      ctaVariant: 'outline',
      footnote: 'Local use. No feature gates.',
      className:
        'col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500',
    },
    {
      title: 'Team',
      price: '€99',
      priceYearly: '€990',
      period: '/month',
      features: [
        'Everything in Home/Lab',
        'Web UI for cluster & models',
        'Shared workspaces & quotas',
        'Priority support (business hours)',
        'Rhai policy templates (rate/data)',
      ],
      ctaText: 'Start 30-Day Trial',
      ctaHref: '/signup?plan=team',
      highlighted: true,
      badge: 'Most Popular',
      footnote: 'Cancel anytime during trial.',
      saveBadge: '2 months free',
      className:
        'col-span-12 md:col-span-4 order-first md:order-none motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-100',
    },
    {
      title: 'Enterprise',
      price: 'Custom',
      features: [
        'Everything in Team',
        'Dedicated, isolated instances',
        'Custom SLAs & onboarding',
        'White-label & SSO options',
        'Enterprise security & support',
      ],
      ctaText: 'Contact Sales',
      ctaHref: '/contact?type=enterprise',
      ctaVariant: 'outline',
      footnote: "We'll reply within 1 business day.",
      className:
        'col-span-12 md:col-span-4 motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:delay-200',
    },
  ],
  footer: {
    mainText: 'Every plan includes the full rbee orchestrator. No feature gates. No artificial limits.',
    subText: 'Prices exclude VAT. OSS license applies to Home/Lab.',
  },
}

// === Testimonials Template ===

/**
 * Testimonials container - Layout configuration
 */
export const testimonialsTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Trusted by Developers Who Value Independence',
  bgVariant: 'background',
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Testimonials Template - Trusted by Developers Who Value Independence
 */
export const testimonialsTemplateProps: TestimonialsTemplateProps = {
  testimonials: [
    {
      avatar: '👨‍💻',
      author: 'Alex K.',
      role: 'Solo Developer',
      quote: 'Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.',
    },
    {
      avatar: '👩‍💼',
      author: 'Sarah M.',
      role: 'CTO',
      quote:
        "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes.",
    },
    {
      avatar: '👨‍🔧',
      author: 'Marcus T.',
      role: 'DevOps Engineer',
      quote: 'Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.',
    },
  ],
  stats: [
    { value: '1,200+', label: 'GitHub stars' },
    { value: '500+', label: 'Active installations' },
    { value: '8,000+', label: 'GPUs orchestrated' },
    { value: '€0', label: 'Avg. monthly cost' },
  ],
}

// === CTA Template ===

/**
 * CTA Template - Stop Depending on AI Providers
 */
export const ctaTemplateProps: CTATemplateProps = {
  title: 'Stop Depending on AI Providers. Start Building Today.',
  subtitle: "Join 500+ developers who've taken control of their AI infrastructure.",
  primary: {
    label: 'Get Started Free',
    href: '/getting-started',
    iconRight: ArrowRight,
  },
  secondary: {
    label: 'View Documentation',
    href: '/docs',
    iconLeft: GitHubIcon,
    variant: 'outline',
  },
  note: '100% open source. No credit card required. Install in 15 minutes.',
}
