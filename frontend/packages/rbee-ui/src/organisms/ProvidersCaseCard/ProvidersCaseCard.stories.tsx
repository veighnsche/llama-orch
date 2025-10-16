import type { Meta, StoryObj } from '@storybook/react'
import { Cpu, Gamepad2, Monitor, Server } from 'lucide-react'
import { ProvidersCaseCard } from './ProvidersCaseCard'

const meta = {
  title: 'Organisms/ProvidersCaseCard',
  component: ProvidersCaseCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersCaseCard>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Gaming PC owner use case
 * Most common provider type with RTX 4090
 */
export const GamingPcOwner: Story = {
  args: {
    icon: <Gamepad2 />,
    title: 'Gaming PC Owners',
    subtitle: 'Most common provider type',
    quote: "I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep.",
    facts: [
      { label: 'Typical GPU:', value: 'RTX 4080–4090' },
      { label: 'Availability:', value: '16–20 h/day' },
      { label: 'Monthly:', value: '€120–180' },
    ],
  },
}

/**
 * Homelab enthusiast use case
 * Multiple GPUs with high earnings
 */
export const HomelabEnthusiast: Story = {
  args: {
    icon: <Server />,
    title: 'Homelab Enthusiasts',
    subtitle: 'Multiple GPUs, high earnings',
    quote: 'Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit.',
    facts: [
      { label: 'Setup:', value: '3–6 GPUs' },
      { label: 'Availability:', value: '20–24 h/day' },
      { label: 'Monthly:', value: '€300–600' },
    ],
  },
}

/**
 * Former crypto miner use case
 * Repurposed mining rigs
 */
export const FormerCryptoMiner: Story = {
  args: {
    icon: <Cpu />,
    title: 'Former Crypto Miners',
    subtitle: 'Repurpose mining rigs',
    quote: 'After PoS, my rig idled. rbee now earns more than mining—with better margins.',
    facts: [
      { label: 'Setup:', value: '6–12 GPUs' },
      { label: 'Availability:', value: '24 h/day' },
      { label: 'Monthly:', value: '€600–1,200' },
    ],
  },
}

/**
 * Workstation owner use case
 * Professional GPUs earning passive income
 */
export const WorkstationOwner: Story = {
  args: {
    icon: <Monitor />,
    title: 'Workstation Owners',
    subtitle: 'Professional GPUs earning',
    quote: 'My RTX 4080 is busy on renders only. The rest of the time it makes ~€100/mo on rbee.',
    facts: [
      { label: 'Typical GPU:', value: 'RTX 4070–4080' },
      { label: 'Availability:', value: '12–16 h/day' },
      { label: 'Monthly:', value: '€80–140' },
    ],
  },
}

/**
 * With highlight badge
 * Shows optional highlight feature
 */
export const WithHighlight: Story = {
  args: {
    icon: <Gamepad2 />,
    title: 'Gaming PC Owners',
    subtitle: 'Most common provider type',
    quote: "I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep.",
    facts: [
      { label: 'Typical GPU:', value: 'RTX 4080–4090' },
      { label: 'Availability:', value: '16–20 h/day' },
      { label: 'Monthly:', value: '€120–180' },
    ],
    highlight: 'Top Earner',
  },
}

/**
 * Minimal facts
 * Shows card with fewer facts
 */
export const MinimalFacts: Story = {
  args: {
    icon: <Server />,
    title: 'Homelab Enthusiasts',
    quote: 'Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit.',
    facts: [
      { label: 'Setup:', value: '3–6 GPUs' },
      { label: 'Monthly:', value: '€300–600' },
    ],
  },
}
