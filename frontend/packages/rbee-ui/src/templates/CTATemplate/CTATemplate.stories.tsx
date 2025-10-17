import { TemplateContainer } from '@rbee/ui/molecules'
import { ctaTemplateProps } from '@rbee/ui/pages/HomePage'
import { educationCTAContainerProps, educationCTAProps } from '@rbee/ui/pages/EducationPage'
import { communityCTAContainerProps, communityCTAProps } from '@rbee/ui/pages/CommunityPage'
import { securityCTAContainerProps, securityCTAProps } from '@rbee/ui/pages/SecurityPage'
import type { Meta, StoryObj } from '@storybook/react'
import { CTATemplate } from './CTATemplate'

const meta = {
  title: 'Templates/CTATemplate',
  component: CTATemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CTATemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnHomeCTA - ctaTemplateProps
 * @tags home, cta, conversion
 *
 * CTATemplate as used on the Home page
 * - "Stop depending on AI providers" headline
 * - Two CTAs: Get started free + View documentation
 * - Gradient emphasis variant
 */
export const OnHomeCTA: Story = {
  args: ctaTemplateProps,
}

/**
 * CTATemplate as used on the Education page
 * - Build real skills CTA
 * - Join students learning distributed AI
 * - Free for education
 */
export const OnEducationCTA: Story = {
  render: (args) => (
    <TemplateContainer {...educationCTAContainerProps}>
      <CTATemplate {...args} />
    </TemplateContainer>
  ),
  args: educationCTAProps,
}

/**
 * CTATemplate as used on the Community page
 * - Community page usage
 */
export const OnCommunityCTA: Story = {
  render: (args) => (
    <TemplateContainer {...communityCTAContainerProps}>
      <CTATemplate {...args} />
    </TemplateContainer>
  ),
  args: communityCTAProps,
}

/**
 * CTATemplate as used on the Security page
 * - Security page usage
 */
export const OnSecurityCTA: Story = {
  render: (args) => (
    <TemplateContainer {...securityCTAContainerProps}>
      <CTATemplate {...args} />
    </TemplateContainer>
  ),
  args: securityCTAProps,
}
