import { enterpriseSecurityProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseSecurity } from './EnterpriseSecurity'

const meta = {
  title: 'Templates/EnterpriseSecurity',
  component: EnterpriseSecurity,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseSecurity>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseSecurity as used on the Enterprise page
 * - Six security crates grid
 * - Security-first architecture
 * - Compliance and audit features
 */
export const OnEnterprisePage: Story = {
  args: enterpriseSecurityProps,
}
