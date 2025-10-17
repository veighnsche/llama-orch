import { TemplateContainer } from '@rbee/ui/molecules'
import { contributionTypesContainerProps, contributionTypesProps } from '@rbee/ui/pages/CommunityPage'
import { educationStudentTypesContainerProps, educationStudentTypesProps } from '@rbee/ui/pages/EducationPage'
import { useCasesTemplateContainerProps, useCasesTemplateProps } from '@rbee/ui/pages/HomePage'
import { useCasesContainerProps, useCasesProps } from '@rbee/ui/pages/ResearchPage'
import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesTemplate } from './UseCasesTemplate'

const meta = {
  title: 'Templates/UseCasesTemplate',
  component: UseCasesTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    columns: {
      control: 'select',
      options: [2, 3],
      description: 'Number of columns in grid',
      table: {
        type: { summary: 'number' },
        defaultValue: { summary: '3' },
      },
    },
  },
} satisfies Meta<typeof UseCasesTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnHomeUseCases - useCasesTemplateProps
 * @tags home, use-cases, personas
 *
 * UseCasesTemplate as used on the Home page
 * - Six use case cards in 3-column grid
 * - Icons: Laptop, Users, HomeIcon, Building, Code, Workflow
 * - Each card has scenario, solution, outcome
 */
export const OnHomeUseCases: Story = {
  render: (args) => (
    <TemplateContainer {...useCasesTemplateContainerProps}>
      <UseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: useCasesTemplateProps,
}

/**
 * UseCasesTemplate as used on the Education page
 * - Student types and learning paths
 * - CS Student, Career Switcher, Researcher
 * - Different goals and outcomes
 */
export const OnEducationStudentTypes: Story = {
  render: (args) => (
    <TemplateContainer {...educationStudentTypesContainerProps}>
      <UseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationStudentTypesProps,
}

/**
 * UseCasesTemplate as used on the Community page
 * - Community page usage
 */
export const OnCommunityUseCases: Story = {
  render: (args) => (
    <TemplateContainer {...contributionTypesContainerProps}>
      <UseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: contributionTypesProps,
}

/**
 * UseCasesTemplate as used on the Research page
 * - Research page usage
 */
export const OnResearchUseCases: Story = {
  render: (args) => (
    <TemplateContainer {...useCasesContainerProps}>
      <UseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: useCasesProps,
}
