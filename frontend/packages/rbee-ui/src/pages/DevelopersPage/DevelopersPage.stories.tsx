import type { Meta, StoryObj } from '@storybook/react'
import DevelopersPage from './DevelopersPage'

const meta = {
  title: 'Pages/DevelopersPage',
  component: DevelopersPage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof DevelopersPage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
