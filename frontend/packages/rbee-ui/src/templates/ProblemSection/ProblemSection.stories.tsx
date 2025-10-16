import type { Meta, StoryObj } from '@storybook/react'
import { problemSectionProps } from '@rbee/ui/pages/HomePage'
import { ProblemSection } from './ProblemSection'

const meta = {
  title: 'Templates/ProblemSection',
  component: ProblemSection,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProblemSection>

export default meta
type Story = StoryObj<typeof meta>

// Use props from HomePage - single source of truth
export const OnHomePage: Story = {
  args: problemSectionProps,
}
