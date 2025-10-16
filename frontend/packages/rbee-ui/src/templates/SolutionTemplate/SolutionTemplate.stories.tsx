import { solutionTemplateProps as developersSolutionProps } from '@rbee/ui/pages/DevelopersPage'
import { solutionTemplateProps } from '@rbee/ui/pages/HomePage'
import { providersSolutionProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { SolutionTemplate } from './SolutionTemplate'

const meta = {
  title: 'Templates/SolutionTemplate',
  component: SolutionTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SolutionTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * SolutionTemplate as used on the Home page
 * - Four benefits with topology visualization
 * - Multi-host BeeArchitecture diagram
 * - Shows CUDA, Metal, and CPU orchestration
 */
export const OnHomePage: Story = {
  args: solutionTemplateProps,
}

/**
 * SolutionTemplate as used on the Developers page
 * - Four benefits focused on developer needs
 * - How It Works steps
 * - OpenAI-compatible API code example in aside
 * - Primary and secondary CTAs
 */
export const OnDevelopersPage: Story = {
  args: developersSolutionProps,
}

/**
 * SolutionTemplate as used on the Providers page
 * - Four benefits focused on GPU providers
 * - How It Works steps for marketplace
 * - Earnings card with GPU pricing estimates
 * - Primary and secondary CTAs
 */
export const OnProvidersPage: Story = {
  args: providersSolutionProps,
}
