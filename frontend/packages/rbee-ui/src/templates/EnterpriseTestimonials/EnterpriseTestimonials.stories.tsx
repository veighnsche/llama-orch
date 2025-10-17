import { enterpriseTestimonialsProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseTestimonials } from './EnterpriseTestimonials'

const meta = {
  title: 'Templates/EnterpriseTestimonials',
  component: EnterpriseTestimonials,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseTestimonials>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseTestimonials as used on the Enterprise page
 * - Regulated industries testimonials
 * - Finance, Healthcare, Legal, Government sectors
 * - Compliance-focused social proof
 */
export const OnEnterprisePage: Story = {
  args: enterpriseTestimonialsProps,
}
