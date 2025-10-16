import type { Meta, StoryObj } from '@storybook/react'
import { ctaTemplateProps } from '@rbee/ui/pages/HomePage'
import { CTATemplate } from './CTATemplate'

const meta = {
  title: 'Templates/CTATemplate',
  component: CTATemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CTATemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * CTATemplate as used on the Home page
 * - "Stop depending on AI providers" headline
 * - Two CTAs: Get started free + View documentation
 * - Gradient emphasis variant
 */
export const OnHomePage: Story = {
  args: ctaTemplateProps,
}
