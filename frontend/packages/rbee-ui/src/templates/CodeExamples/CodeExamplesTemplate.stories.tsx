import type { Meta, StoryObj } from '@storybook/react'
import { codeExamplesProps } from '@rbee/ui/pages/DevelopersPage'
import { CodeExamplesTemplate } from './CodeExamplesTemplate'

const meta = {
  title: 'Templates/CodeExamplesTemplate',
  component: CodeExamplesTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CodeExamplesTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * CodeExamplesTemplate as used on the Developers page
 * - Interactive code examples with tabbed navigation
 * - Syntax highlighting and copy-to-clipboard
 * - Keyboard navigation support
 */
export const OnDevelopersPage: Story = {
  args: codeExamplesProps,
}
