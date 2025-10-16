import type { Meta, StoryObj } from '@storybook/react'
import { faqBeehive } from '@rbee/ui/assets'
import { FAQTemplate } from './FAQTemplate'

const meta = {
  title: 'Templates/FAQTemplate',
  component: FAQTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FAQTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    badgeText: 'Support • Self-hosted AI',
    categories: ['Setup', 'Models', 'Performance', 'Marketplace', 'Security', 'Production'],
    faqItems: [
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
            <p>We're in M0 today—great for dev and homelabs. Production SLAs, health monitoring, and marketplace land across M1–M3.</p>
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
    ],
    supportCard: {
      image: faqBeehive,
      imageAlt:
        'Isometric illustration of a vibrant community hub: hexagonal beehive structure with worker bees collaborating around glowing question mark icons, speech bubbles floating between honeycomb cells containing miniature server racks, warm amber and honey-gold palette with soft cyan accents, friendly bees wearing tiny headsets offering support, knowledge base documents scattered on wooden surface, gentle directional lighting creating welcoming atmosphere, detailed technical diagrams visible through translucent honeycomb walls, community-driven support concept, approachable and helpful mood',
      title: 'Still stuck?',
      links: [
        {
          label: 'Join Discussions',
          href: 'https://github.com/veighnsche/llama-orch/discussions',
        },
        { label: 'Read Setup Guide', href: '/docs/setup' },
        { label: 'Email support', href: 'mailto:support@rbee.dev' },
      ],
      cta: {
        label: 'Open Discussions',
        href: 'https://github.com/veighnsche/llama-orch/discussions',
      },
    },
    jsonLdEnabled: true,
  },
}
