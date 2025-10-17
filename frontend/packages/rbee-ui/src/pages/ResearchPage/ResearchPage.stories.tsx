import type { Meta, StoryObj } from '@storybook/react'
import ResearchPage from './ResearchPage'

const meta = {
  title: 'Pages/ResearchPage',
  component: ResearchPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Research page targeting academic researchers and ML scientists. Showcases reproducible experiments, deterministic seeds, audit trails, and research-grade infrastructure.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ResearchPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Research page with all sections:
 * - Hero with research infrastructure overview
 * - Reproducibility features
 * - Deterministic experiments
 * - Audit trail and provenance
 * - Research use cases
 * - Academic partnerships
 * - Publications and citations
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}
