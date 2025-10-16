import type { Meta, StoryObj } from '@storybook/react'
import { SecurityIsolationTemplate } from './SecurityIsolationTemplate'
import { securityIsolationProps } from '@rbee/ui/pages/FeaturesPage'

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
