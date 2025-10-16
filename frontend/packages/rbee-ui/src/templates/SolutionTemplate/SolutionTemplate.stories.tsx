import { solutionTemplateProps } from '@rbee/ui/pages/HomePage'
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

// Use props from HomePage - single source of truth
export const OnHomePage: Story = {
  args: solutionTemplateProps,
}
