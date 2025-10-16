import type { Meta, StoryObj } from '@storybook/react'
import { EarningsCard } from './EarningsCard'

const meta = {
  title: 'Molecules/EarningsCard',
  component: EarningsCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EarningsCard>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default earnings card with GPU pricing estimates
 * As used on the Providers page
 */
export const Default: Story = {
  args: {
    title: 'Estimated Monthly Earnings',
    rows: [
      {
        model: 'RTX 4090',
        meta: '24GB VRAM • 450W',
        value: '€180/mo',
        note: 'at 80% utilization',
      },
      {
        model: 'RTX 4080',
        meta: '16GB VRAM • 320W',
        value: '€140/mo',
        note: 'at 80% utilization',
      },
      {
        model: 'RTX 3080',
        meta: '10GB VRAM • 320W',
        value: '€90/mo',
        note: 'at 80% utilization',
      },
    ],
    disclaimer: 'Actuals vary with demand, pricing, and availability. These are conservative estimates.',
  },
}

/**
 * Compliance metrics variant
 * Shows different use case with compliance-focused data
 */
export const ComplianceMetrics: Story = {
  args: {
    title: 'Compliance Metrics',
    rows: [
      {
        model: 'GDPR Compliance',
        meta: 'Data residency',
        value: '100%',
        note: 'EU servers only',
      },
      {
        model: 'SOC 2 Type II',
        meta: 'Security audit',
        value: 'Certified',
        note: 'Annual review',
      },
      {
        model: 'ISO 27001',
        meta: 'Information security',
        value: 'Certified',
        note: 'Valid until 2026',
      },
    ],
  },
}

/**
 * Performance metrics variant
 * Shows performance-focused data without disclaimer
 */
export const PerformanceMetrics: Story = {
  args: {
    title: 'Performance Metrics',
    rows: [
      {
        model: 'Llama 3.1 70B',
        meta: 'Tokens per second',
        value: '45 t/s',
      },
      {
        model: 'Mistral 7B',
        meta: 'Tokens per second',
        value: '120 t/s',
      },
      {
        model: 'CodeLlama 34B',
        meta: 'Tokens per second',
        value: '68 t/s',
      },
    ],
  },
}

/**
 * Minimal variant
 * Shows minimal configuration with just two rows
 */
export const Minimal: Story = {
  args: {
    rows: [
      {
        model: 'Active Workers',
        meta: 'Across network',
        value: '12',
      },
      {
        model: 'Total VRAM',
        meta: 'Available',
        value: '192 GB',
      },
    ],
  },
}
