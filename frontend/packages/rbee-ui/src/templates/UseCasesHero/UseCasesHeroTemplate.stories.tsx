import type { Meta, StoryObj } from '@storybook/react'
import { useCasesHero } from '@rbee/ui/assets'
import { UseCasesHeroTemplate } from './UseCasesHeroTemplate'

const meta = {
  title: 'Templates/UseCasesHeroTemplate',
  component: UseCasesHeroTemplate,
  parameters: { layout: 'fullscreen' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnUseCasesPage: Story = {
  args: {
    badgeText: 'OpenAI-compatible',
    heading: 'Built for Those Who Value',
    headingHighlight: 'Independence',
    description:
      'Own your AI infrastructure. From solo developers to enterprises, rbee adapts to your needs without compromising power or flexibility.',
    primaryCta: {
      text: 'Explore use cases',
      href: '#use-cases',
    },
    secondaryCta: {
      text: 'See architecture',
      href: '#architecture',
    },
    proofIndicators: [
      { text: 'Self-hosted', hasDot: true },
      { text: 'OpenAI-compatible' },
      { text: 'CUDA · Metal · CPU' },
    ],
    image: useCasesHero,
    imageAlt: 'Homelab setup with AI inference running on local hardware',
    imageCaption: 'Your models, your hardware — no lock-in.',
  },
}
