import type { Meta, StoryObj } from '@storybook/react'
import { providersCTAProps } from '@rbee/ui/pages/ProvidersPage'
import { ProvidersCTA } from './ProvidersCTA'

const meta = {
  title: 'Templates/ProvidersCTA',
  component: ProvidersCTA,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersCTA>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersCTA as used on the Providers page
 * - Final call-to-action for GPU providers
 * - Encourages signup and onboarding
 * - Features clear next steps and benefits
 */
export const OnProvidersPage: Story = {
  args: providersCTAProps,
}
