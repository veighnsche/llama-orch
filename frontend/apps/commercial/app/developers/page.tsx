import { Alert, AlertDescription, Card, CardContent, CardHeader, CardTitle, GitHubIcon } from '@rbee/ui/atoms'
import { CodeBlock } from '@rbee/ui/molecules'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import {
  CTASection,
  DevelopersCodeExamples,
  DevelopersHero,
  EmailCapture,
  HowItWorksSection,
  PricingSection,
  SolutionSection,
  TestimonialsSection,
  UseCasesSection,
} from '@rbee/ui/organisms'
import { CoreFeaturesTabs } from '@rbee/ui/organisms/CoreFeaturesTabs'
import { ProblemSection } from '@rbee/ui/organisms/ProblemSection'
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

export default function DevelopersPage() {
  return (
    <main className="min-h-screen bg-slate-950">
      <DevelopersHero />
      <EmailCapture />
      <ProblemSection
        id="risk"
        kicker="The Hidden Cost of Dependency"
        title="The Hidden Risk of AI-Assisted Development"
        subtitle="You're building complex codebases with AI assistance. But what happens when your provider changes the rules?"
        items={[
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
        ]}
        ctaPrimary={{ label: 'Take Control', href: '/getting-started' }}
        ctaSecondary={{ label: 'View Documentation', href: '/docs' }}
        ctaCopy="Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers."
      />
      <SolutionSection
        id="how-it-works"
        kicker="How rbee Works"
        title="Your Hardware. Your Models. Your Control."
        subtitle="rbee orchestrates AI inference across every device in your home network, turning idle hardware into a private, OpenAI-compatible AI platform."
        features={[
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
        ]}
        steps={[
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
        ]}
        aside={
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
                <AlertDescription className="text-xs">
                  Works with Cursor, Zed, Continue, and any OpenAI SDK
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        }
        ctaPrimary={{
          label: 'Get Started',
          href: '/getting-started',
        }}
        ctaSecondary={{
          label: 'View Documentation',
          href: '/docs',
        }}
      />
      <HowItWorksSection
        id="quickstart"
        title="From zero to AI infrastructure in 15 minutes"
        steps={[
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
        ]}
      />
      <CoreFeaturesTabs
        title="Core capabilities"
        description="Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time."
        tabs={[
          {
            value: 'api',
            icon: Code,
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
            benefits: [
              { text: 'No vendor lock-in' },
              { text: 'Use your models + GPUs' },
              { text: 'Keep existing tooling' },
            ],
          },
          {
            value: 'gpu',
            icon: Cpu,
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
            icon: Gauge,
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
            benefits: [
              { text: 'Deterministic routing' },
              { text: 'Policy & compliance ready' },
              { text: 'Easy to evolve' },
            ],
          },
          {
            value: 'sse',
            icon: Zap,
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
        ]}
        defaultTab="api"
      />
      <UseCasesSection
        id="use-cases"
        title="Built for developers who value independence"
        items={[
          {
            icon: Code,
            title: 'Build your own AI coder',
            scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
            solution: 'Run rbee on a gaming PC + old workstation. Llama-3.1-70B for code; Stable Diffusion for assets.',
            outcome: '$0/month AI costs. Full control. No rate limits.',
            tags: ['OpenAI-compatible', 'Local models'],
          },
          {
            icon: FileText,
            title: 'Documentation generators',
            scenario: 'Need comprehensive docs from codebase; API costs are prohibitive.',
            solution: 'Process entire repos locally with rbee. Generate markdown with examples.',
            outcome: 'Process entire repos. Zero API costs. Private by default.',
            tags: ['Markdown', 'Privacy'],
          },
          {
            icon: FlaskConical,
            title: 'Test generators',
            scenario: 'Writing tests is time-consuming; need AI to generate comprehensive suites.',
            solution: 'Use rbee + llama-orch-utils to generate Jest/Vitest tests from specs.',
            outcome: '10× faster coverage. No external dependencies.',
            tags: ['Jest', 'Vitest'],
          },
          {
            icon: GitPullRequest,
            title: 'Code review agents',
            scenario: 'Small team needs automated code review but cannot afford enterprise tools.',
            solution: 'Build custom review agent with rbee. Analyze PRs for issues, security, performance.',
            outcome: 'Automated reviews. Zero ongoing costs. Custom rules.',
            tags: ['GitHub', 'GitLab'],
          },
          {
            icon: Wrench,
            title: 'Refactoring agents',
            scenario: 'Legacy codebase needs modernization; manual refactoring would take months.',
            solution: 'Use rbee to refactor code to modern patterns. TypeScript, async/await, etc.',
            outcome: 'Months of work → days. You approve every change.',
            tags: ['TypeScript', 'Modernization'],
          },
        ]}
      />
      <DevelopersCodeExamples />
      <PricingSection variant="home" showKicker={false} showEditorialImage={false} />
      <TestimonialsSection
        title="Trusted by Developers Who Value Independence"
        testimonials={[
          {
            avatar: '👨‍💻',
            author: 'Alex K.',
            role: 'Solo Developer',
            quote:
              'Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.',
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
        ]}
        stats={[
          { value: '1,200+', label: 'GitHub stars' },
          { value: '500+', label: 'Active installations' },
          { value: '8,000+', label: 'GPUs orchestrated' },
          { value: '€0', label: 'Avg. monthly cost' },
        ]}
      />
      <CTASection
        title="Stop Depending on AI Providers. Start Building Today."
        subtitle="Join 500+ developers who've taken control of their AI infrastructure."
        primary={{
          label: 'Get Started Free',
          href: '/getting-started',
          iconRight: ArrowRight,
        }}
        secondary={{
          label: 'View Documentation',
          href: '/docs',
          iconLeft: GitHubIcon,
          variant: 'outline',
        }}
        note="100% open source. No credit card required. Install in 15 minutes."
      />
    </main>
  )
}
