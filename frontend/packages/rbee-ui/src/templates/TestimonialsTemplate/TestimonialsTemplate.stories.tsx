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

export const OnHomePage: Story = {
  args: {
    testimonials: [
      {
        avatar: '👨‍💻',
        author: 'Alex K.',
        role: 'Solo Developer',
        quote:
          'Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.',
      },
      {
        avatar: '👩‍💼',
        author: 'Sarah M.',
        role: 'CTO',
        quote:
          "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes.",
      },
      {
        avatar: '👨‍🔧',
        author: 'Marcus T.',
        role: 'DevOps',
        quote: 'Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.',
      },
    ],
    stats: [
      { value: '1,200+', label: 'GitHub stars', valueTone: 'foreground' },
      {
        value: '500+',
        label: 'Active installations',
        valueTone: 'foreground',
      },
      {
        value: '8,000+',
        label: 'GPUs orchestrated',
        valueTone: 'foreground',
      },
      { value: '€0', label: 'Avg. monthly cost', valueTone: 'primary' },
    ],
  },
}
