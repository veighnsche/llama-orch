import type { Meta, StoryObj } from '@storybook/react'
import { problemTemplateProps } from '@rbee/ui/pages/HomePage'
import { ProblemTemplate } from './ProblemTemplate'

const meta = {
  title: 'Templates/ProblemTemplate',
  component: ProblemTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProblemTemplate>

export default meta
type Story = StoryObj<typeof meta>

// Use props from HomePage - single source of truth
export const OnHomePage: Story = {
  args: problemTemplateProps,
}
