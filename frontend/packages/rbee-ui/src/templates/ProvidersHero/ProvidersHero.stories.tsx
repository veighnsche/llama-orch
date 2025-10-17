import { providersHeroProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersHero } from './ProvidersHero'

const meta = {
  title: 'Templates/ProvidersHero',
  component: ProvidersHero,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersHero>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnProvidersHero - providersHeroProps
 * @tags providers, hero, gpu, earnings
 * 
 * ProvidersHero as used on the Providers page
 * - GPU provider-focused hero messaging
 * - Emphasizes earning potential and marketplace benefits
 * - Features provider-specific CTAs
 */
export const OnProvidersHero: Story = {
  args: providersHeroProps,
}
