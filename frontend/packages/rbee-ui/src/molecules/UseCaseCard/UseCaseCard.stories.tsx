import type { Meta, StoryObj } from '@storybook/react'
import { Code, FileText, MessageSquare, Search } from 'lucide-react'
import { UseCaseCard } from './UseCaseCard'

const meta: Meta<typeof UseCaseCard> = {
  title: 'Molecules/UseCaseCard',
  component: UseCaseCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The UseCaseCard molecule displays a use case with scenario, solution, and highlights. Features icon, badge, and scroll-to-anchor support.

## Composition
This molecule is composed of:
- **IconBox**: Use case icon with color
- **Badge**: Optional badge (e.g., "Popular", "New")
- **Title**: Use case name
- **Scenario**: Problem description
- **Solution**: How rbee solves it
- **Highlights**: Key benefits with checkmarks

## When to Use
- Use case pages
- Feature demonstrations
- Solution showcases
- Customer success stories

## Used In
- **UseCasesPrimary**: Displays primary use cases (chatbots, document analysis, code generation, semantic search)
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Lucide icon component',
      table: {
        type: { summary: 'LucideIcon' },
        category: 'Content',
      },
    },
    iconTone: {
      control: 'select',
      options: ['primary', 'muted', 'success', 'warning'],
      description: 'Icon tone',
      table: {
        type: { summary: "'primary' | 'muted' | 'success' | 'warning'" },
        category: 'Appearance',
      },
    },
    title: {
      control: 'text',
      description: 'Use case title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    scenario: {
      control: 'text',
      description: 'Problem/scenario description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    solution: {
      control: 'text',
      description: 'How rbee solves it',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    tags: {
      control: 'object',
      description: 'Tag labels',
      table: {
        type: { summary: 'string[]' },
        category: 'Content',
      },
    },
    anchor: {
      control: 'text',
      description: 'Optional anchor ID',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
    cta: {
      control: 'object',
      description: 'Optional CTA link',
      table: {
        type: { summary: '{ label: string; href: string }' },
        category: 'Content',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof UseCaseCard>

export const Default: Story = {
  args: {
    icon: MessageSquare,
    iconTone: 'primary',
    title: 'Customer Support Chatbots',
    scenario: 'Your support team is overwhelmed with repetitive questions, and cloud AI services expose customer data.',
    solution:
      'Deploy a private chatbot that handles common queries 24/7 while keeping all conversations on your infrastructure.',
    tags: ['24/7 Support', 'Private', 'Scalable'],
  },
}

export const WithIcon: Story = {
  args: {
    icon: FileText,
    iconTone: 'primary',
    title: 'Document Analysis',
    scenario: 'Legal teams spend hours reviewing contracts and compliance documents manually.',
    solution: 'Automate document review with AI that understands legal language and flags potential issues.',
    tags: ['Legal', 'Compliance', 'Fast'],
  },
}

export const WithOutcome: Story = {
  args: {
    icon: Code,
    iconTone: 'primary',
    title: 'Code Generation & Review',
    scenario: "Developers need AI assistance but can't send proprietary code to external services.",
    solution: 'Run code-aware LLMs on your infrastructure to assist with development without exposing IP.',
    outcome: 'Developers get AI assistance without exposing proprietary code.',
    tags: ['Development', 'Security', 'Multi-language'],
    anchor: 'code-generation',
  },
}

export const InUseCasesContext: Story = {
  render: () => (
    <div className="w-full max-w-6xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: UseCaseCard in UseCasesPrimary organism</div>
      <div className="grid gap-6 md:grid-cols-2">
        <UseCaseCard
          icon={MessageSquare}
          iconTone="primary"
          title="Customer Support Chatbots"
          scenario="Your support team is overwhelmed with repetitive questions, and cloud AI services expose customer data."
          solution="Deploy a private chatbot that handles common queries 24/7 while keeping all conversations on your infrastructure."
          tags={['24/7 Support', 'Private', 'Scalable']}
          anchor="chatbots"
        />
        <UseCaseCard
          icon={Search}
          iconTone="primary"
          title="Semantic Search"
          scenario="Traditional keyword search misses relevant documents and frustrates users."
          solution="Implement AI-powered semantic search that understands meaning, not just keywords."
          tags={['Search', 'Multilingual', 'Private']}
          anchor="search"
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'UseCaseCard as used in the UseCasesPrimary organism, showing two primary use cases.',
      },
    },
  },
}
