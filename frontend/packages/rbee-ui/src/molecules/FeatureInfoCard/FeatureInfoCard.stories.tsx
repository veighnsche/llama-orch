import type { Meta, StoryObj } from '@storybook/react'
import { DollarSign, Lock, Shield, Zap } from 'lucide-react'
import { FeatureInfoCard } from './FeatureInfoCard'

const meta = {
  title: 'Molecules/FeatureInfoCard',
  component: FeatureInfoCard,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FeatureInfoCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    icon: <DollarSign />,
    title: 'Zero ongoing costs',
    body: 'Pay only for electricity. No API bills, no per-token surprises.',
    tone: 'default',
    size: 'sm',
  },
}

export const Neutral: Story = {
  args: {
    icon: <Zap />,
    title: 'Neutral background',
    body: 'Clean, minimal design with neutral background for solution sections.',
    tone: 'neutral',
    size: 'sm',
  },
}

export const Primary: Story = {
  args: {
    icon: <Zap />,
    title: 'Complete privacy',
    body: 'Code and data never leave your network. Audit-ready by design.',
    tone: 'primary',
    size: 'sm',
  },
}

export const Destructive: Story = {
  args: {
    icon: <Lock />,
    title: 'The provider shuts down',
    body: 'APIs get deprecated. Your AI-built code becomes unmaintainable overnight.',
    tone: 'destructive',
    size: 'base',
    tag: 'Loss €2,400/mo',
  },
}

export const Muted: Story = {
  args: {
    icon: <Shield />,
    title: 'Security first',
    body: 'Built with security best practices from the ground up.',
    tone: 'muted',
    size: 'sm',
  },
}

export const WithTag: Story = {
  args: {
    icon: <DollarSign />,
    title: 'The price increases',
    body: '$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.',
    tone: 'destructive',
    size: 'base',
    tag: 'Loss €50/mo',
  },
}

export const LargeText: Story = {
  args: {
    icon: <Lock />,
    title: 'Larger body text',
    body: 'This card uses size="base" for larger, more readable body text. Perfect for problem sections.',
    tone: 'destructive',
    size: 'base',
  },
}

export const SolutionsGrid = {
  render: () => (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
      <FeatureInfoCard
        icon={<DollarSign />}
        title="Zero ongoing costs"
        body="Pay only for electricity. No API bills, no per-token surprises."
        tone="neutral"
        size="sm"
      />
      <FeatureInfoCard
        icon={<Lock />}
        title="Complete privacy"
        body="Code and data never leave your network. Audit-ready by design."
        tone="neutral"
        size="sm"
      />
      <FeatureInfoCard
        icon={<Zap />}
        title="Locked to your rules"
        body="Models update only when you approve. No breaking changes."
        tone="neutral"
        size="sm"
      />
      <FeatureInfoCard
        icon={<Shield />}
        title="Use all your hardware"
        body="CUDA, Metal, and CPU orchestrated as one pool."
        tone="neutral"
        size="sm"
      />
    </div>
  ),
}

export const ProblemsGrid = {
  render: () => (
    <div className="grid gap-6 sm:gap-7 md:grid-cols-3">
      <FeatureInfoCard
        icon={<Lock />}
        title="The model changes"
        body="Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked."
        tone="destructive"
        size="base"
        delay="delay-75"
      />
      <FeatureInfoCard
        icon={<DollarSign />}
        title="The price increases"
        body="$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral."
        tone="destructive"
        size="base"
        tag="Loss €50/mo"
        delay="delay-150"
      />
      <FeatureInfoCard
        icon={<Shield />}
        title="The provider shuts down"
        body="APIs get deprecated. Your AI-built code becomes unmaintainable overnight."
        tone="destructive"
        size="base"
        delay="delay-200"
      />
    </div>
  ),
}
