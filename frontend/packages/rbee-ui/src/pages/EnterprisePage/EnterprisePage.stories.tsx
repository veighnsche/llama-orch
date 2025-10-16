import type { Meta, StoryObj } from '@storybook/react'
import EnterprisePage from './EnterprisePage'

const meta = {
  title: 'Pages/EnterprisePage',
  component: EnterprisePage,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterprisePage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
