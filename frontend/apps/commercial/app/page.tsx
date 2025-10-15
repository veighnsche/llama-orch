import {
  AudienceSelector,
  ComparisonSection,
  CTASection,
  DefaultCoreFeaturesTabs,
  EmailCapture,
  FAQSection,
  HeroSection,
  HomeSolutionSection,
  HowItWorksSection,
  PricingSection,
  ProblemSection,
  TechnicalSection,
  TestimonialsSection,
  UseCasesSection,
  WhatIsRbee,
} from '@rbee/ui/organisms'
import { faqBeehive } from '@rbee/ui/assets'
import {
  AlertTriangle,
  Anchor,
  ArrowRight,
  BookOpen,
  Building,
  Code,
  DollarSign,
  Home as HomeIcon,
  Laptop,
  Lock,
  Shield,
  Users,
  Workflow,
} from 'lucide-react'

export default function Home() {
  return (
    <main className="min-h-screen">
      <HeroSection />
      <WhatIsRbee />
      <AudienceSelector />
      <EmailCapture />
      <ProblemSection
        title="The hidden risk of AI-assisted development"
        subtitle="You're building complex codebases with AI assistance. What happens when the provider changes the rules?"
        items={[
          {
            title: 'The model changes',
            body: 'Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked.',
            icon: <AlertTriangle className="h-6 w-6" />,
            tone: 'destructive',
          },
          {
            title: 'The price increases',
            body: '$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.',
            icon: <DollarSign className="h-6 w-6" />,
            tone: 'primary',
          },
          {
            title: 'The provider shuts down',
            body: 'APIs get deprecated. Your AI-built code becomes unmaintainable overnight.',
            icon: <Lock className="h-6 w-6" />,
            tone: 'destructive',
          },
        ]}
      />
      <HomeSolutionSection
        title="Your hardware. Your models. Your control."
        subtitle="rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform."
        benefits={[
          {
            icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Zero ongoing costs',
            body: 'Pay only for electricity. No API bills, no per-token surprises.',
          },
          {
            icon: <Shield className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Complete privacy',
            body: 'Code and data never leave your network. Audit-ready by design.',
          },
          {
            icon: <Anchor className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Locked to your rules',
            body: 'Models update only when you approve. No breaking changes.',
          },
          {
            icon: <Laptop className="h-6 w-6 text-primary" aria-hidden="true" />,
            title: 'Use all your hardware',
            body: 'CUDA, Metal, and CPU orchestrated as one pool.',
          },
        ]}
        topology={{
          mode: 'multi-host',
          hosts: [
            {
              hostLabel: 'Gaming PC',
              workers: [
                { id: 'w0', label: 'GPU 0', kind: 'cuda' },
                { id: 'w1', label: 'GPU 1', kind: 'cuda' },
              ],
            },
            {
              hostLabel: 'MacBook Pro',
              workers: [{ id: 'w2', label: 'GPU 0', kind: 'metal' }],
            },
            {
              hostLabel: 'Workstation',
              workers: [
                { id: 'w3', label: 'GPU 0', kind: 'cuda' },
                { id: 'w4', label: 'CPU 0', kind: 'cpu' },
              ],
            },
          ],
        }}
      />
      <HowItWorksSection
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
                  <div className="text-[var(--syntax-comment)]">rbee-keeper daemon start</div>
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
                  <div className="text-[var(--syntax-comment)]">
                    rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20
                  </div>
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
                    <span className="text-[var(--syntax-keyword)]">export</span>{' '}
                    OPENAI_API_BASE=http://localhost:8080/v1
                  </div>
                  <div className="text-[var(--syntax-comment)]">
                    # OpenAI-compatible endpoint — works with Zed & Cursor
                  </div>
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
              lines: (
                <>
                  <div>
                    <span className="text-[var(--syntax-import)]">import</span> {'{'} invoke {'}'}{' '}
                    <span className="text-[var(--syntax-import)]">from</span>{' '}
                    <span className="text-[var(--syntax-string)]">&apos;@rbee/utils&apos;</span>;
                  </div>
                  <div className="mt-2">
                    <span className="text-[var(--syntax-keyword)]">const</span> code ={' '}
                    <span className="text-[var(--syntax-keyword)]">await</span>{' '}
                    <span className="text-[var(--syntax-function)]">invoke</span>
                    {'({'}
                  </div>
                  <div className="pl-4">
                    prompt: <span className="text-[var(--syntax-string)]">&apos;Generate API from schema&apos;</span>,
                  </div>
                  <div className="pl-4">
                    model: <span className="text-[var(--syntax-string)]">&apos;llama-3.1-70b&apos;</span>
                  </div>
                  <div>{'});'}</div>
                </>
              ),
              copyText:
                "import { invoke } from '@rbee/utils';\n\nconst code = await invoke({\n  prompt: 'Generate API from schema',\n  model: 'llama-3.1-70b'\n});",
            },
          },
        ]}
      />
      <DefaultCoreFeaturesTabs />
      <UseCasesSection
        title="Built for those who value independence"
        subtitle="Run serious AI on your own hardware. Keep costs at zero, keep control at 100%."
        items={[
          {
            icon: Laptop,
            title: 'The solo developer',
            scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
            solution:
              'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
            outcome: '$0/month AI costs. Full control. No rate limits.',
          },
          {
            icon: Users,
            title: 'The small team',
            scenario: '5-person startup burning $500/mo on APIs.',
            solution:
              'Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.',
            outcome: '$6,000+ saved per year. GDPR-friendly by design.',
          },
          {
            icon: HomeIcon,
            title: 'The homelab enthusiast',
            scenario: 'Four GPUs gathering dust.',
            solution: 'Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.',
            outcome: 'Idle GPUs → productive. Auto-download models, clean shutdowns.',
          },
          {
            icon: Building,
            title: 'The enterprise',
            scenario: '50-dev org. Code cannot leave the premises.',
            solution:
              'On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.',
            outcome: 'EU-only compliance. Zero external dependencies.',
          },
          {
            icon: Code,
            title: 'The AI-dependent coder',
            scenario:
              'Building complex codebases with Claude/GPT-4. Fears provider changes, shutdowns, or price hikes.',
            solution:
              'Build your own AI coders with rbee + llama-orch-utils. OpenAI-compatible API runs on YOUR hardware.',
            outcome: 'Complete independence. Models never change without permission. $0/month forever.',
          },
          {
            icon: Workflow,
            title: 'The agentic AI builder',
            scenario: 'Needs to build custom AI agents: code generators, doc writers, test creators, code reviewers.',
            solution:
              'Use llama-orch-utils TypeScript library: file ops, LLM invocation, prompt management, response extraction.',
            outcome:
              'Build production AI agents in hours. Full control. No rate limits. Test reproducibility built-in.',
          },
        ]}
      />
      <ComparisonSection />
      <PricingSection />
      <TestimonialsSection
        title="Trusted by developers who value independence"
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
            role: 'DevOps',
            quote: 'Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.',
          },
        ]}
        stats={[
          { value: '1,200+', label: 'GitHub stars', valueTone: 'foreground' },
          {
            value: '500+',
            label: 'Active installations',
            valueTone: 'foreground',
          },
          {
            value: '8,000+',
            label: 'GPUs orchestrated',
            valueTone: 'foreground',
          },
          { value: '€0', label: 'Avg. monthly cost', valueTone: 'primary' },
        ]}
      />
      <TechnicalSection />
      <FAQSection
        title="rbee FAQ"
        subtitle="Quick answers about setup, models, orchestration, and security."
        badgeText="Support • Self-hosted AI"
        categories={['Setup', 'Models', 'Performance', 'Marketplace', 'Security', 'Production']}
        faqItems={[
          {
            value: 'item-1',
            question: 'How is this different from Ollama?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>
                  Ollama excels on a single machine. rbee orchestrates across machines and backends (CUDA, Metal, CPU),
                  with an OpenAI-compatible, task-based API and SSE streaming—plus a programmable scheduler and optional
                  marketplace federation.
                </p>
              </div>
            ),
            category: 'Performance',
          },
          {
            value: 'item-2',
            question: 'Do I need to be a Rust expert?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>No. Use prebuilt binaries via CLI or Web UI. Customize routing with simple Rhai scripts or YAML if needed.</p>
              </div>
            ),
            category: 'Setup',
          },
          {
            value: 'item-3',
            question: "What if I don't have GPUs?",
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>CPU-only works (slower). You can later federate to external GPU providers via the marketplace.</p>
              </div>
            ),
            category: 'Setup',
          },
          {
            value: 'item-4',
            question: 'Is this production-ready?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>
                  We're in M0 today—great for dev and homelabs. Production SLAs, health monitoring, and marketplace land
                  across M1–M3.
                </p>
              </div>
            ),
            category: 'Production',
          },
          {
            value: 'item-5',
            question: 'How do I migrate from OpenAI API?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>
                  Switch one env var: <code>export OPENAI_API_BASE=http://localhost:8080/v1</code>
                </p>
              </div>
            ),
            category: 'Setup',
          },
          {
            value: 'item-6',
            question: 'What models are supported?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>Any GGUF from Hugging Face (Llama, Mistral, Qwen, DeepSeek). Image gen and TTS arrive in M2.</p>
              </div>
            ),
            category: 'Models',
          },
          {
            value: 'item-7',
            question: 'Can I sell GPU time?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>Yes—via the marketplace in M3: register your node and earn from excess capacity.</p>
              </div>
            ),
            category: 'Marketplace',
          },
          {
            value: 'item-8',
            question: 'What about security?',
            answer: (
              <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
                <p>
                  Runs entirely on your network. Rhai scripts are sandboxed (time & memory limits). Platform mode uses
                  immutable schedulers for multi-tenant isolation.
                </p>
              </div>
            ),
            category: 'Security',
          },
        ]}
        showSupportCard={true}
        supportCardImage={faqBeehive}
        supportCardImageAlt="Isometric illustration of a vibrant community hub: hexagonal beehive structure with worker bees collaborating around glowing question mark icons, speech bubbles floating between honeycomb cells containing miniature server racks, warm amber and honey-gold palette with soft cyan accents, friendly bees wearing tiny headsets offering support, knowledge base documents scattered on wooden surface, gentle directional lighting creating welcoming atmosphere, detailed technical diagrams visible through translucent honeycomb walls, community-driven support concept, approachable and helpful mood"
        supportCardTitle="Still stuck?"
        supportCardLinks={[
          { label: 'Join Discussions', href: 'https://github.com/yourusername/rbee/discussions' },
          { label: 'Read Setup Guide', href: '/docs/setup' },
          { label: 'Email support', href: 'mailto:support@example.com' },
        ]}
        supportCardCTA={{ label: 'Open Discussions', href: 'https://github.com/yourusername/rbee/discussions' }}
        jsonLdEnabled={true}
      />
      <CTASection
        title="Stop depending on AI providers. Start building today."
        subtitle="Join 500+ developers who've taken control of their AI infrastructure."
        primary={{
          label: 'Get started free',
          href: '/getting-started',
          iconRight: ArrowRight,
        }}
        secondary={{
          label: 'View documentation',
          href: '/docs',
          iconLeft: BookOpen,
          variant: 'outline',
        }}
        note="100% open source. No credit card required. Install in 15 minutes."
        emphasis="gradient"
      />
    </main>
  )
}
