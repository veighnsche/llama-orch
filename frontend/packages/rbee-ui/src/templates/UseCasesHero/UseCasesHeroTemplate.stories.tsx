import type { Meta, StoryObj } from '@storybook/react'
import { useCasesHeroProps } from '@rbee/ui/pages'
import { UseCasesHeroTemplate } from './UseCasesHeroTemplate'

const meta = {
  title: 'Templates/UseCasesHeroTemplate',
  component: UseCasesHeroTemplate,
  parameters: { layout: 'fullscreen' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * UseCasesHeroTemplate as used on the Use Cases page
 * - "Built for Those Who Value Independence" headline
 * - OpenAI-compatible badge
 * - Three proof indicators (Self-hosted, OpenAI-compatible, CUDA · Metal · CPU)
 * - Homelab hero image
 */
export const OnUseCasesPage: Story = {
  args: useCasesHeroProps,
}
