import type { Meta, StoryObj } from '@storybook/react'
import { emailCaptureProps } from '@rbee/ui/pages/HomePage'
import { EmailCapture } from './EmailCapture'

const meta = {
  title: 'Templates/EmailCapture',
  component: EmailCapture,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EmailCapture>

export default meta
type Story = StoryObj<typeof meta>

// Use props from HomePage - single source of truth
export const OnHomePage: Story = {
  args: emailCaptureProps,
}
