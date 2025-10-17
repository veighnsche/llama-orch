import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { devopsErrorHandlingContainerProps, devopsErrorHandlingProps } from '../../pages/DevOpsPage'
import { errorHandlingContainerProps, errorHandlingProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { securityErrorHandlingContainerProps, securityErrorHandlingProps } from '../../pages/SecurityPage'
import { ErrorHandlingTemplate } from './ErrorHandlingTemplate'

const meta: Meta<typeof ErrorHandlingTemplate> = {
  title: 'Templates/ErrorHandlingTemplate',
  component: ErrorHandlingTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ErrorHandlingTemplate>

/**
 * OnFeaturesErrorHandling - errorHandlingProps
 * @tags features, error-handling, resilience, playbook
 *
 * General error handling and resilience patterns
 */
export const OnFeaturesErrorHandling: Story = {
  render: (args) => (
    <TemplateContainer {...errorHandlingContainerProps}>
      <ErrorHandlingTemplate {...args} />
    </TemplateContainer>
  ),
  args: errorHandlingProps,
}

/**
 * OnDevOpsErrorHandling - devopsErrorHandlingProps
 * @tags devops, error-handling, monitoring, recovery
 *
 * DevOps-focused error handling with automatic recovery and monitoring
 */
export const OnDevOpsErrorHandling: Story = {
  render: (args) => (
    <TemplateContainer {...devopsErrorHandlingContainerProps}>
      <ErrorHandlingTemplate {...args} />
    </TemplateContainer>
  ),
  args: devopsErrorHandlingProps,
}

/**
 * OnSecurityErrorHandling - securityErrorHandlingProps
 * @tags security, threat-detection, incident-response, audit
 *
 * Security-focused error handling with threat detection and audit trails
 */
export const OnSecurityErrorHandling: Story = {
  render: (args) => (
    <TemplateContainer {...securityErrorHandlingContainerProps}>
      <ErrorHandlingTemplate {...args} />
    </TemplateContainer>
  ),
  args: securityErrorHandlingProps,
}
