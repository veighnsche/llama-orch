import type { Meta, StoryObj } from '@storybook/react'
import { testimonialsTemplateProps } from '@rbee/ui/pages/HomePage'
import { TestimonialsTemplate } from './TestimonialsTemplate'

const meta = {
  title: 'Templates/TestimonialsTemplate',
  component: TestimonialsTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TestimonialsTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * TestimonialsTemplate as used on the Home page
 * - Three testimonials (Alex K., Sarah M., Marcus T.)
 * - Four stats (GitHub stars, installations, GPUs, cost)
 * - Emoji avatars
 */
export const OnHomePage: Story = {
  args: testimonialsTemplateProps,
}
