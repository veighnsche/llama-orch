import type { Meta, StoryObj } from '@storybook/react'
import { providersHeroProps } from '@rbee/ui/pages/ProvidersPage'
import { ProvidersHeroTemplate } from './ProvidersHeroTemplate'

const meta = {
  title: 'Templates/ProvidersHeroTemplate',
  component: ProvidersHeroTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersHeroTemplate as used on the Providers page
 * - GPU provider-focused hero messaging
 * - Emphasizes earning potential and marketplace benefits
 * - Features provider-specific CTAs
 */
export const OnProvidersPage: Story = {
  args: providersHeroProps,
}
