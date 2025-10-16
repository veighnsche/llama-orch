import type { Meta, StoryObj } from '@storybook/react'
import { useCasesPrimaryProps } from '@rbee/ui/pages'
import { UseCasesPrimaryTemplate } from './UseCasesPrimaryTemplate'

const meta = {
  title: 'Templates/UseCasesPrimaryTemplate',
  component: UseCasesPrimaryTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesPrimaryTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * UseCasesPrimaryTemplate as used on the Use Cases page
 * - Eight use case cards (Solo Developer, Small Team, Homelab, Enterprise, etc.)
 * - Filter tabs (All, Solo, Team, Enterprise, Research)
 * - UsecasesGridDark hero image
 * - Each card has icon, scenario, solution, outcome, and highlights
 */
export const OnUseCasesPage: Story = {
  args: useCasesPrimaryProps,
}
