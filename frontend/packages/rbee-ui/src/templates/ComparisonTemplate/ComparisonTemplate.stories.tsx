import { comparisonTemplateProps } from '@rbee/ui/pages/HomePage'
import type { Meta, StoryObj } from '@storybook/react'
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

/**
 * ComparisonTemplate as used on the Home page
 * - Four columns: rbee, OpenAI & Anthropic, Ollama, Runpod & Vast.ai
 * - Six comparison rows (Cost, Privacy, Multi-GPU, API, Routing, Rate Limits)
 * - Legend with Check/X icons
 * - Two CTAs at bottom
 */
export const OnHomePage: Story = {
  args: comparisonTemplateProps,
}
