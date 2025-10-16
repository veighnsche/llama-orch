import type { Meta, StoryObj } from '@storybook/react'
import UseCasesPage from './UseCasesPage'

const meta = {
  title: 'Pages/UseCasesPage',
  component: UseCasesPage,
  parameters: { layout: 'fullscreen' },
} satisfies Meta<typeof UseCasesPage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
