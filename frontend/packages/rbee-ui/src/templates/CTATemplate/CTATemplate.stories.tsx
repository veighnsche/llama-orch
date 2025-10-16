import type { Meta, StoryObj } from '@storybook/react'
import { ArrowRight, BookOpen } from 'lucide-react'
import { CTATemplate } from './CTATemplate'

const meta = {
  title: 'Templates/CTATemplate',
  component: CTATemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CTATemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    title: 'Stop depending on AI providers. Start building today.',
    subtitle: "Join 500+ developers who've taken control of their AI infrastructure.",
    primary: {
      label: 'Get started free',
      href: '/getting-started',
      iconRight: ArrowRight,
    },
    secondary: {
      label: 'View documentation',
      href: '/docs',
      iconLeft: BookOpen,
      variant: 'outline',
    },
    note: '100% open source. No credit card required. Install in 15 minutes.',
    emphasis: 'gradient',
  },
}
