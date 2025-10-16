import type { Meta, StoryObj } from '@storybook/react'
import { providersCTAProps } from '@rbee/ui/pages/ProvidersPage'
import { ProvidersCTATemplate } from './ProvidersCTATemplate'

const meta = {
  title: 'Templates/ProvidersCTATemplate',
  component: ProvidersCTATemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersCTATemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersCTATemplate as used on the Providers page
 * - Final call-to-action for GPU providers
 * - Encourages signup and onboarding
 * - Features clear next steps and benefits
 */
export const OnProvidersPage: Story = {
  args: providersCTAProps,
}
