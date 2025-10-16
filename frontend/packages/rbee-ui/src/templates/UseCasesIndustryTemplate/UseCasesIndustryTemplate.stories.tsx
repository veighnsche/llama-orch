import { useCasesIndustryProps } from '@rbee/ui/pages/UseCasesPage/UseCasesPageProps'
import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesIndustryTemplate } from './UseCasesIndustryTemplate'

const meta = {
  title: 'Templates/UseCasesIndustryTemplate',
  component: UseCasesIndustryTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesIndustryTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * UseCasesIndustryTemplate as used on the Use Cases page
 * - Six industry cards (Financial Services, Healthcare, Legal, Government, Education, Manufacturing)
 * - Filter tabs for each industry
 * - IndustriesHero image
 * - Compliance badges (GDPR, HIPAA, ITAR, FERPA)
 */
export const OnUseCasesPage: Story = {
  args: useCasesIndustryProps,
}
