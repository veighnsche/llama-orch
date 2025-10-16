import type { Meta, StoryObj } from '@storybook/react'
import { Check, X } from 'lucide-react'
import { ComparisonTemplate } from './ComparisonTemplate'

const meta = {
  title: 'Templates/ComparisonTemplate',
  component: ComparisonTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ComparisonTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    columns: [
      { key: 'rbee', label: 'rbee', accent: true },
      { key: 'openai', label: 'OpenAI & Anthropic' },
      { key: 'ollama', label: 'Ollama' },
      { key: 'runpod', label: 'Runpod & Vast.ai' },
    ],
    rows: [
      {
        feature: 'Total Cost',
        values: {
          rbee: '$0 (runs on your hardware)',
          openai: '$20–100/mo per dev',
          ollama: '$0',
          runpod: '$0.50–2/hr',
        },
      },
      {
        feature: 'Privacy / Data Residency',
        values: {
          rbee: true,
          openai: false,
          ollama: true,
          runpod: false,
        },
        note: 'Complete data control vs. limited',
      },
      {
        feature: 'Multi-GPU Utilization',
        values: {
          rbee: true,
          openai: 'N/A',
          ollama: 'Limited',
          runpod: true,
        },
      },
      {
        feature: 'OpenAI-Compatible API',
        values: {
          rbee: true,
          openai: true,
          ollama: 'Partial',
          runpod: false,
        },
      },
      {
        feature: 'Custom Routing Policies',
        values: {
          rbee: true,
          openai: false,
          ollama: false,
          runpod: false,
        },
      },
      {
        feature: 'Rate Limits / Quotas',
        values: {
          rbee: 'None',
          openai: 'Yes',
          ollama: 'None',
          runpod: 'Yes',
        },
      },
    ],
    legend: [
      {
        icon: <Check className="h-3.5 w-3.5 text-chart-3" aria-hidden="true" />,
        label: 'Available',
      },
      {
        icon: <X className="h-3.5 w-3.5 text-destructive" aria-hidden="true" />,
        label: 'Not available',
      },
    ],
    legendNote: '"Partial" = limited coverage',
    footerMessage: 'Bring your own GPUs, keep your data in-house.',
    ctas: [
      { label: 'See Quickstart', href: '/docs/quickstart' },
      { label: 'Architecture', href: '/docs/architecture', variant: 'ghost' },
    ],
  },
}
