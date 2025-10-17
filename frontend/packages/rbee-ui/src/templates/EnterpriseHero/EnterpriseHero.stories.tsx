import { enterpriseHeroProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseHero } from './EnterpriseHero'

const meta = {
  title: 'Templates/EnterpriseHero',
  component: EnterpriseHero,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseHero>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnEnterpriseHero - enterpriseHeroProps
 * @tags enterprise, hero, gdpr, compliance, eu
 * 
 * EnterpriseHero as used on the Enterprise page
 * - GDPR-compliant messaging
 * - Immutable audit trail console visual
 * - Compliance chips (GDPR, SOC2, ISO 27001)
 * - EU data residency badges
 */
export const OnEnterpriseHero: Story = {
  args: enterpriseHeroProps,
}
