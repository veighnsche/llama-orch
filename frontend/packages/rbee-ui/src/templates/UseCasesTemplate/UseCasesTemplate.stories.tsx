import type { Meta, StoryObj } from '@storybook/react'
import { useCasesTemplateProps } from '@rbee/ui/pages/HomePage'
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
 * UseCasesTemplate as used on the Home page
 * - Six use case cards in 3-column grid
 * - Icons: Laptop, Users, HomeIcon, Building, Code, Workflow
 * - Each card has scenario, solution, outcome
 */
export const OnHomePage: Story = {
  args: useCasesTemplateProps,
}
