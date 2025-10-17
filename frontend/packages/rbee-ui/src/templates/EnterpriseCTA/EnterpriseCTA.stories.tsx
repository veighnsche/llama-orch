import { enterpriseCTAProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseCTA } from './EnterpriseCTA'

const meta = {
  title: 'Templates/EnterpriseCTA',
  component: EnterpriseCTA,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseCTA>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseCTA as used on the Enterprise page
 * - Three CTA options (Schedule Demo, Compliance Pack, Talk to Sales)
 * - Eyebrow labels for context
 * - Trust signals and response time commitments
 */
export const OnEnterprisePage: Story = {
  args: enterpriseCTAProps,
}
