import type { Meta, StoryObj } from '@storybook/react'
import { FAQSection, type FAQItem } from './FaqSection'

const defaultCategories = ['Setup', 'Models', 'Performance', 'Marketplace', 'Security', 'Production']

const defaultFaqItems: FAQItem[] = [
  {
    value: 'item-1',
    question: 'How is this different from Ollama?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:bg-muted prose-code:rounded">
        <p>
          Ollama excels on a single machine. rbee orchestrates across machines and backends (CUDA, Metal, CPU), with an
          OpenAI-compatible, task-based API and SSE streaming—plus a programmable scheduler and optional marketplace
          federation.
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
          We're in M0 today—great for dev and homelabs. Production SLAs, health monitoring, and marketplace land across
          M1–M3.
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
          Runs entirely on your network. Rhai scripts are sandboxed (time & memory limits). Platform mode uses immutable
          schedulers for multi-tenant isolation.
        </p>
      </div>
    ),
    category: 'Security',
  },
]

const meta = {
  title: 'Organisms/Home/FAQSection',
  component: FAQSection,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The FAQSection provides an interactive, searchable FAQ interface with category filtering, accordion-style answers, and an optional support card. It includes JSON-LD structured data for SEO and features a comprehensive search and filter system.

## Composition
This organism contains:
- **Header**: Badge, title, and subtitle
- **Search Bar**: Real-time search across questions and answers
- **Category Filters**: Clickable pills to filter by category
- **Expand/Collapse Controls**: Buttons to manage accordion state
- **Accordion**: Collapsible Q&A items grouped by category
- **Support Card**: Optional sidebar with image, links, and CTA
- **JSON-LD Schema**: Structured data for search engines

## When to Use
- On the homepage to address common questions
- On dedicated FAQ/support pages
- After pricing sections to address objections
- In documentation to supplement technical content

## Content Requirements
- **FAQ Items**: Questions with detailed answers (8-12 recommended)
- **Categories**: Logical groupings (Setup, Models, Performance, etc.)
- **Search**: Works across question text and answer content
- **Support Card**: Links to additional resources

## Variants
- **Default**: Full layout with support card
- **Without Support Card**: FAQ list only
- **Filtered**: Pre-filtered by category
- **Custom Content**: Override default FAQs

## Examples
\`\`\`tsx
import { FAQSection } from '@rbee/ui/organisms/FaqSection'

// Default with all features
<FAQSection />

// Without support card
<FAQSection showSupportCard={false} />

// Custom title and subtitle
<FAQSection
  title="Common Questions"
  subtitle="Everything you need to know about rbee"
  badgeText="Help Center"
/>

// Custom FAQ items
<FAQSection
  faqItems={[
    {
      value: 'custom-1',
      question: 'How do I get started?',
      answer: <p>Download rbee and run the installer...</p>,
      category: 'Setup'
    }
  ]}
/>
\`\`\`

## Used In
- Home page (/)
- Pricing page (/pricing)
- Support/FAQ page

## Related Components
- Accordion
- Badge
- Button
- Input
- Separator

## Accessibility
- **Keyboard Navigation**: Full keyboard support for search, filters, and accordion
- **ARIA**: Proper labels on search input and accordion items
- **Focus Management**: Visible focus indicators throughout
- **Screen Readers**: Accordion state announced properly
- **Search**: Live region for search results count
- **JSON-LD**: Structured data for search engines
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Section title',
      table: {
        type: { summary: 'string' },
        defaultValue: { summary: 'rbee FAQ' },
        category: 'Content',
      },
    },
    subtitle: {
      control: 'text',
      description: 'Section subtitle',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    badgeText: {
      control: 'text',
      description: 'Badge text above title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    showSupportCard: {
      control: 'boolean',
      description: 'Show support card in sidebar',
      table: {
        type: { summary: 'boolean' },
        defaultValue: { summary: 'true' },
        category: 'Appearance',
      },
    },
    jsonLdEnabled: {
      control: 'boolean',
      description: 'Enable JSON-LD structured data',
      table: {
        type: { summary: 'boolean' },
        defaultValue: { summary: 'true' },
        category: 'SEO',
      },
    },
  },
} satisfies Meta<typeof FAQSection>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  parameters: {
    docs: {
      description: {
        story:
          'Default FAQ section with all features: search, category filters, accordion, and support card. Use the theme toggle in the toolbar to switch between light and dark modes.',
      },
    },
  },
}

export const WithoutSupportCard: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: false,
    jsonLdEnabled: true,
  },
  parameters: {
    docs: {
      description: {
        story:
          'FAQ section without the support card sidebar, useful for narrower layouts or when support resources are elsewhere.',
      },
    },
  },
}

export const CustomContent: Story = {
  args: {
    title: 'Common Questions',
    subtitle: 'Find answers to frequently asked questions about rbee',
    badgeText: 'Help Center',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  parameters: {
    docs: {
      description: {
        story: 'FAQ section with custom title, subtitle, and badge text.',
      },
    },
  },
}

export const InteractiveSearch: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  render: () => (
    <div>
      <FAQSection
        title="rbee FAQ"
        subtitle="Quick answers about setup, models, orchestration, and security."
        badgeText="Support • Self-hosted AI"
        categories={defaultCategories}
        faqItems={defaultFaqItems}
        showSupportCard={true}
        jsonLdEnabled={true}
      />
      <div style={{ padding: '2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Try the Search</h2>
        <ol
          style={{
            listStyle: 'decimal',
            paddingLeft: '2rem',
            textAlign: 'left',
            maxWidth: '600px',
            margin: '0 auto',
            lineHeight: '1.8',
          }}
        >
          <li>Type keywords in the search bar (e.g., "Ollama", "GPU", "models")</li>
          <li>Watch the FAQ list filter in real-time</li>
          <li>Click category buttons to filter by topic</li>
          <li>Use "Expand all" / "Collapse all" to manage accordion state</li>
          <li>Click questions to expand/collapse answers</li>
        </ol>
        <p style={{ marginTop: '1rem', color: '#666' }}>Search works across both questions and answers.</p>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Interactive demo of the search and filter functionality.',
      },
    },
  },
}

export const CategoryFiltering: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  render: () => (
    <div>
      <FAQSection
        title="rbee FAQ"
        subtitle="Quick answers about setup, models, orchestration, and security."
        badgeText="Support • Self-hosted AI"
        categories={defaultCategories}
        faqItems={defaultFaqItems}
        showSupportCard={true}
        jsonLdEnabled={true}
      />
      <div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
          Category Organization
        </h2>
        <div
          style={{
            maxWidth: '800px',
            margin: '0 auto',
            display: 'grid',
            gap: '1rem',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          }}
        >
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Setup</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>Installation, configuration, getting started</p>
          </div>
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Models</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>Supported models, formats, compatibility</p>
          </div>
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Performance</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>Optimization, scaling, benchmarks</p>
          </div>
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Marketplace</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>GPU sharing, earning, federation</p>
          </div>
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Security</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>Privacy, isolation, sandboxing</p>
          </div>
          <div>
            <h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Production</h3>
            <p style={{ fontSize: '0.875rem', color: '#666' }}>SLAs, monitoring, enterprise features</p>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Overview of FAQ categories and their content focus.',
      },
    },
  },
}

export const SupportCardHighlight: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  render: () => (
    <div>
      <FAQSection
        title="rbee FAQ"
        subtitle="Quick answers about setup, models, orchestration, and security."
        badgeText="Support • Self-hosted AI"
        categories={defaultCategories}
        faqItems={defaultFaqItems}
        showSupportCard={true}
        jsonLdEnabled={true}
      />
      <div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
          Support Resources
        </h2>
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <p style={{ marginBottom: '1rem' }}>The support card (visible on desktop) provides quick access to:</p>
          <ul style={{ listStyle: 'disc', paddingLeft: '2rem', lineHeight: '1.8' }}>
            <li>
              <strong>GitHub Discussions</strong>: Ask questions and get help from the community
            </li>
            <li>
              <strong>Setup Guide</strong>: Step-by-step documentation for getting started
            </li>
            <li>
              <strong>Email Support</strong>: Direct contact for urgent issues
            </li>
          </ul>
          <p style={{ marginTop: '1rem', color: '#666' }}>
            The card is sticky-positioned for easy access while scrolling.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Highlights the support card and its resources.',
      },
    },
  },
}

export const SEOFeatures: Story = {
  args: {
    title: 'rbee FAQ',
    subtitle: 'Quick answers about setup, models, orchestration, and security.',
    badgeText: 'Support • Self-hosted AI',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: true,
    jsonLdEnabled: true,
  },
  render: () => (
    <div>
      <FAQSection
        title="rbee FAQ"
        subtitle="Quick answers about setup, models, orchestration, and security."
        badgeText="Support • Self-hosted AI"
        categories={defaultCategories}
        faqItems={defaultFaqItems}
        showSupportCard={true}
        jsonLdEnabled={true}
      />
      <div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
          SEO Features
        </h2>
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>JSON-LD Structured Data</h3>
          <p style={{ marginBottom: '1rem', color: '#666' }}>
            The FAQ section includes FAQPage schema markup for search engines. This helps Google display rich snippets
            in search results.
          </p>

          <h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Semantic HTML</h3>
          <p style={{ marginBottom: '1rem', color: '#666' }}>
            Uses proper semantic elements: &lt;section&gt;, &lt;nav&gt;, &lt;aside&gt;, and heading hierarchy.
          </p>

          <h3 style={{ fontWeight: 'bold', marginTop: '1rem', marginBottom: '0.5rem' }}>Accessibility</h3>
          <p style={{ color: '#666' }}>
            ARIA labels, keyboard navigation, and screen reader support ensure the FAQ is accessible to all users.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Overview of SEO and accessibility features built into the FAQ section.',
      },
    },
  },
}

// Created by: TEAM-004
export const PricingPageVariant: Story = {
  args: {
    title: 'Pricing FAQs',
    subtitle: 'Answers on licensing, upgrades, trials, and payments.',
    badgeText: 'Pricing • Plans & Billing',
    categories: defaultCategories,
    faqItems: defaultFaqItems,
    showSupportCard: false,
    jsonLdEnabled: true,
  },
  parameters: {
    docs: {
      description: {
        story:
          'Pricing page variant of FAQ section with custom title, subtitle, and badge. This variant is used on /pricing page with pricing-specific FAQs addressing: "Is it really free?", "What\'s included in Pro vs. Enterprise?", "Can I upgrade/downgrade?", "What payment methods?", "Refund policy?". Support card is hidden (showSupportCard={false}) as pricing page has its own CTAs. Custom categories would include: Licensing, Plans & Tiers, Billing & Payments, Upgrades & Downgrades.',
      },
    },
  },
}
