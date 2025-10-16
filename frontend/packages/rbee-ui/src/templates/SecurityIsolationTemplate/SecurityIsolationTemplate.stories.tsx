import type { Meta, StoryObj } from '@storybook/react'
import { securityIsolationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { SecurityIsolationTemplate } from './SecurityIsolationTemplate'

const meta: Meta<typeof SecurityIsolationTemplate> = {
  title: 'Templates/SecurityIsolationTemplate',
  component: SecurityIsolationTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof SecurityIsolationTemplate>

export const OnFeaturesPage: Story = {
  args: securityIsolationProps,
}
