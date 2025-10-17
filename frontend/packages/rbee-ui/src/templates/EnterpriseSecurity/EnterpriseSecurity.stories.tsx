import { enterpriseSecurityContainerProps, enterpriseSecurityProps } from '@rbee/ui/pages/EnterprisePage'
import { TemplateContainer } from '@rbee/ui/molecules'
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
 * OnEnterpriseSecurity - enterpriseSecurityProps
 * @tags enterprise, security, defense
 * 
 * EnterpriseSecurity as used on the Enterprise page
 * - Six security crates grid
 * - Security-first architecture
 * - Compliance and audit features
 */
export const OnEnterpriseSecurity: Story = {
  args: enterpriseSecurityProps,
  render: (args) => (
    <TemplateContainer {...enterpriseSecurityContainerProps}>
      <EnterpriseSecurity {...args} />
    </TemplateContainer>
  ),
}
