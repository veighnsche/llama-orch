import { enterpriseSolutionProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseSolution } from './EnterpriseSolution'

const meta = {
  title: 'Templates/EnterpriseSolution',
  component: EnterpriseSolution,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseSolution>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseSolution as used on the Enterprise page
 * - How rbee Works section
 * - Features, steps, and metrics
 * - Enterprise value proposition
 */
export const OnEnterprisePage: Story = {
  args: enterpriseSolutionProps,
}
