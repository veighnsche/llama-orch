import { useCasesTemplateContainerProps, useCasesTemplateProps } from '@rbee/ui/pages/HomePage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesTemplate } from './UseCasesTemplate'

const meta = {
  title: 'Templates/UseCasesTemplate',
  component: UseCasesTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    columns: {
      control: 'select',
      options: [2, 3],
      description: 'Number of columns in grid',
      table: {
        type: { summary: 'number' },
        defaultValue: { summary: '3' },
      },
    },
  },
} satisfies Meta<typeof UseCasesTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnHomeUseCases - useCasesTemplateProps
 * @tags home, use-cases, personas
 * 
 * UseCasesTemplate as used on the Home page
 * - Six use case cards in 3-column grid
 * - Icons: Laptop, Users, HomeIcon, Building, Code, Workflow
 * - Each card has scenario, solution, outcome
 */
export const OnHomeUseCases: Story = {
  render: (args) => (
    <TemplateContainer {...useCasesTemplateContainerProps}>
      <UseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: useCasesTemplateProps,
}
