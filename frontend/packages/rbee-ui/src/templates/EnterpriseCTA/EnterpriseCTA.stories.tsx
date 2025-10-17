import { TemplateContainer } from '@rbee/ui/molecules'
import { enterpriseCTAContainerProps, enterpriseCTAProps } from '@rbee/ui/pages/EnterprisePage'
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
 * OnEnterpriseCTA - enterpriseCTAProps
 * @tags enterprise, cta, demo
 *
 * EnterpriseCTA as used on the Enterprise page
 * - Three CTA options (Schedule Demo, Compliance Pack, Talk to Sales)
 * - Eyebrow labels for context
 * - Trust signals and response time commitments
 * - Wrapped in TemplateContainer with title and description
 */
export const OnEnterpriseCTA: Story = {
  args: enterpriseCTAProps,
  render: (args) => (
    <TemplateContainer {...enterpriseCTAContainerProps}>
      <EnterpriseCTA {...args} />
    </TemplateContainer>
  ),
}
