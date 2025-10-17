import { TemplateContainer } from '@rbee/ui/molecules'
import { enterpriseSecurityContainerProps, enterpriseSecurityProps } from '@rbee/ui/pages/EnterprisePage'
import { educationCurriculumContainerProps, educationCurriculumProps } from '@rbee/ui/pages/EducationPage'
import { complianceSecurityContainerProps, complianceSecurityProps } from '@rbee/ui/pages/CompliancePage'
import { devopsSecurityContainerProps, devopsSecurityProps } from '@rbee/ui/pages/DevOpsPage'
import { securityCratesContainerProps, securityCratesProps } from '@rbee/ui/pages/SecurityPage'
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

/**
 * EnterpriseSecurity as used on the Education page
 * - Six core curriculum modules
 * - Foundations to Production
 * - Comprehensive coverage
 */
export const OnEducationCurriculum: Story = {
  render: (args) => (
    <TemplateContainer {...educationCurriculumContainerProps}>
      <EnterpriseSecurity {...args} />
    </TemplateContainer>
  ),
  args: educationCurriculumProps,
}

/**
 * EnterpriseSecurity as used on the Compliance page
 * - Compliance page usage
 */
export const OnComplianceSecurity: Story = {
  render: (args) => (
    <TemplateContainer {...complianceSecurityContainerProps}>
      <EnterpriseSecurity {...args} />
    </TemplateContainer>
  ),
  args: complianceSecurityProps,
}

/**
 * EnterpriseSecurity as used on the DevOps page
 * - DevOps page usage
 */
export const OnDevOpsSecurity: Story = {
  render: (args) => (
    <TemplateContainer {...devopsSecurityContainerProps}>
      <EnterpriseSecurity {...args} />
    </TemplateContainer>
  ),
  args: devopsSecurityProps,
}

/**
 * EnterpriseSecurity as used on the Security page
 * - Security page usage
 */
export const OnSecuritySecurity: Story = {
  render: (args) => (
    <TemplateContainer {...securityCratesContainerProps}>
      <EnterpriseSecurity {...args} />
    </TemplateContainer>
  ),
  args: securityCratesProps,
}
