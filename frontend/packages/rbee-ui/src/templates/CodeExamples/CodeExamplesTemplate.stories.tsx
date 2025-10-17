import { codeExamplesContainerProps, codeExamplesProps } from '@rbee/ui/pages/DevelopersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
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
 * OnDevelopersCodeExamples - codeExamplesProps
 * @tags developers, code, examples
 * 
 * CodeExamplesTemplate as used on the Developers page
 * - Interactive code examples with tabbed navigation
 * - Syntax highlighting and copy-to-clipboard
 * - Keyboard navigation support
 */
export const OnDevelopersCodeExamples: Story = {
  render: (args) => (
    <TemplateContainer {...codeExamplesContainerProps}>
      <CodeExamplesTemplate {...args} />
    </TemplateContainer>
  ),
  args: codeExamplesProps,
}
