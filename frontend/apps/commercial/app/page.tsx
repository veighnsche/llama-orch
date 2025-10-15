import {
  AudienceSelector,
  ComparisonSection,
  CTASection,
  EmailCapture,
  FAQSection,
  FeaturesSection,
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
      <FeaturesSection />
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
      <FAQSection />
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
