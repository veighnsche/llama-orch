import type { Meta, StoryObj } from '@storybook/react'
import StartupsPage from './StartupsPage'

const meta = {
  title: 'Pages/StartupsPage',
  component: StartupsPage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof StartupsPage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
