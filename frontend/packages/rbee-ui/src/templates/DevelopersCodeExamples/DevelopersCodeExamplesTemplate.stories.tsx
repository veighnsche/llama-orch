import type { Meta, StoryObj } from '@storybook/react'
import { developersCodeExamplesProps } from '@rbee/ui/pages/DevelopersPage'
import { DevelopersCodeExamplesTemplate } from './DevelopersCodeExamplesTemplate'

const meta = {
  title: 'Templates/DevelopersCodeExamplesTemplate',
  component: DevelopersCodeExamplesTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevelopersCodeExamplesTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * DevelopersCodeExamplesTemplate as used on the Developers page
 * - Interactive code examples with syntax highlighting
 * - Multiple language support (Python, TypeScript, cURL)
 * - Copy-to-clipboard functionality
 */
export const OnDevelopersPage: Story = {
  args: developersCodeExamplesProps,
}
