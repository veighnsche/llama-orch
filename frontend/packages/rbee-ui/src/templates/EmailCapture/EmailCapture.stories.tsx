import { TemplateContainer } from '@rbee/ui/molecules'
import { communityEmailCaptureContainerProps, communityEmailCaptureProps } from '@rbee/ui/pages/CommunityPage'
import { complianceEmailCaptureContainerProps, complianceEmailCaptureProps } from '@rbee/ui/pages/CompliancePage'
import { developersEmailCaptureContainerProps, developersEmailCaptureProps } from '@rbee/ui/pages/DevelopersPage'
import { devopsEmailCaptureContainerProps, devopsEmailCaptureProps } from '@rbee/ui/pages/DevOpsPage'
import { educationEmailCaptureContainerProps, educationEmailCaptureProps } from '@rbee/ui/pages/EducationPage'
import { enterpriseEmailCaptureContainerProps, enterpriseEmailCaptureProps } from '@rbee/ui/pages/EnterprisePage'
import { featuresEmailCaptureContainerProps, featuresEmailCaptureProps } from '@rbee/ui/pages/FeaturesPage'
import { emailCaptureContainerProps, emailCaptureProps } from '@rbee/ui/pages/HomePage'
import { pricingEmailCaptureContainerProps, pricingEmailCaptureProps } from '@rbee/ui/pages/PricingPage'
import {
  emailCaptureContainerProps as researchEmailCaptureContainerProps,
  emailCaptureProps as researchEmailCaptureProps,
} from '@rbee/ui/pages/ResearchPage'
import { securityEmailCaptureContainerProps, securityEmailCaptureProps } from '@rbee/ui/pages/SecurityPage'
import { useCasesEmailCaptureContainerProps, useCasesEmailCaptureProps } from '@rbee/ui/pages/UseCasesPage'
import type { Meta, StoryObj } from '@storybook/react'
import { EmailCapture } from './EmailCapture'

const meta = {
  title: 'Templates/EmailCapture',
  component: EmailCapture,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EmailCapture>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EmailCapture as used on the Home page
 * - Badge with development status and pulse
 * - Community-focused messaging
 * - GitHub and Discord links
 * - Homelab bee illustration
 */
export const OnHomePage: Story = {
  render: (args) => (
    <TemplateContainer {...emailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: emailCaptureProps,
}

/**
 * EmailCapture as used on the Features page
 * - Focus on feature updates
 * - Performance improvements messaging
 * - Community highlights
 */
export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...featuresEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: featuresEmailCaptureProps,
}

/**
 * EmailCapture as used on the Use Cases page
 * - Focus on use cases and best practices
 * - Simpler messaging without badge
 */
export const OnUseCasesPage: Story = {
  render: (args) => (
    <TemplateContainer {...useCasesEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: useCasesEmailCaptureProps,
}

/**
 * EmailCapture as used on the Pricing page
 * - Focus on pricing updates
 * - Plans and billing messaging
 * - Discount notifications
 */
export const OnPricingPage: Story = {
  render: (args) => (
    <TemplateContainer {...pricingEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: pricingEmailCaptureProps,
}

/**
 * EmailCapture as used on the Developers page
 * - Developer-focused messaging
 * - Build AI tools without vendor lock-in
 * - GitHub and Discord community links
 */
export const OnDevelopersPage: Story = {
  render: (args) => (
    <TemplateContainer {...developersEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: developersEmailCaptureProps,
}

/**
 * EmailCapture as used on the Enterprise page
 * - Enterprise-focused messaging
 * - GDPR-compliant AI infrastructure
 * - Compliance and security focus
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: enterpriseEmailCaptureProps,
}

/**
 * EmailCapture as used on the Education page
 * - Educator resources focus
 * - Curriculum guides and teaching materials
 * - Free for educators messaging
 */
export const OnEducationPage: Story = {
  render: (args) => (
    <TemplateContainer {...educationEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: educationEmailCaptureProps,
}

/**
 * EmailCapture as used on the Community page
 * - Community page usage
 */
export const OnCommunityPage: Story = {
  render: (args) => (
    <TemplateContainer {...communityEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: communityEmailCaptureProps,
}

/**
 * EmailCapture as used on the Compliance page
 * - Compliance page usage
 */
export const OnCompliancePage: Story = {
  render: (args) => (
    <TemplateContainer {...complianceEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: complianceEmailCaptureProps,
}

/**
 * EmailCapture as used on the DevOps page
 * - DevOps page usage
 */
export const OnDevOpsPage: Story = {
  render: (args) => (
    <TemplateContainer {...devopsEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: devopsEmailCaptureProps,
}

/**
 * EmailCapture as used on the Research page
 * - Research page usage
 */
export const OnResearchPage: Story = {
  render: (args) => (
    <TemplateContainer {...researchEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: researchEmailCaptureProps,
}

/**
 * EmailCapture as used on the Security page
 * - Security page usage
 */
export const OnSecurityPage: Story = {
  render: (args) => (
    <TemplateContainer {...securityEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: securityEmailCaptureProps,
}
