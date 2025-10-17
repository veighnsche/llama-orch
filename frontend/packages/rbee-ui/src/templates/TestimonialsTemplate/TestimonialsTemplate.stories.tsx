import { TemplateContainer } from '@rbee/ui/molecules'
import { testimonialsTemplateContainerProps, testimonialsTemplateProps } from '@rbee/ui/pages/HomePage'
import { educationTestimonialsContainerProps, educationTestimonialsData } from '@rbee/ui/pages/EducationPage'
import { communityStatsContainerProps, communityStatsProps } from '@rbee/ui/pages/CommunityPage'
import type { Meta, StoryObj } from '@storybook/react'
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
 * OnHomeTestimonials - testimonialsTemplateProps
 * @tags home, testimonials, social-proof
 *
 * TestimonialsTemplate as used on the Home page
 * - Three testimonials (Alex K., Sarah M., Marcus T.)
 * - Four stats (GitHub stars, installations, GPUs, cost)
 * - Emoji avatars
 */
export const OnHomeTestimonials: Story = {
  render: (args) => (
    <TemplateContainer {...testimonialsTemplateContainerProps}>
      <TestimonialsTemplate {...args} />
    </TemplateContainer>
  ),
  args: testimonialsTemplateProps,
}

/**
 * TestimonialsTemplate as used on the Education page
 * - Student success stories
 * - Real outcomes and job placements
 * - Portfolio projects
 */
export const OnEducationTestimonials: Story = {
  render: (args) => (
    <TemplateContainer {...educationTestimonialsContainerProps}>
      <TestimonialsTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationTestimonialsData,
}

/**
 * TestimonialsTemplate as used on the Community page
 * - Community page usage
 */
export const OnCommunityTestimonials: Story = {
  render: (args) => (
    <TemplateContainer {...communityStatsContainerProps}>
      <TestimonialsTemplate {...args} />
    </TemplateContainer>
  ),
  args: communityStatsProps,
}
