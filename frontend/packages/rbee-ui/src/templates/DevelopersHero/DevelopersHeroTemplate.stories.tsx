import type { Meta, StoryObj } from '@storybook/react'
import { developersHeroProps } from '@rbee/ui/pages/DevelopersPage'
import { DevelopersHeroTemplate } from './DevelopersHeroTemplate'

const meta = {
  title: 'Templates/DevelopersHeroTemplate',
  component: DevelopersHeroTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevelopersHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * DevelopersHeroTemplate as used on the Developers page
 * - Developer-focused hero with code-first messaging
 * - Emphasizes API simplicity and flexibility
 * - Features terminal-style code examples
 */
export const OnDevelopersPage: Story = {
  args: developersHeroProps,
}
