import { ProvidersSecurityCard } from '@rbee/ui/molecules'
import { ProvidersCaseCard } from '@rbee/ui/organisms'
import type { Meta, StoryObj } from '@storybook/react'
import { Cpu, Eye, FileCheck, Gamepad2, Lock, Monitor, Server, Shield } from 'lucide-react'
import { CardGridTemplate } from './CardGridTemplate'

const meta = {
  title: 'Templates/CardGridTemplate',
  component: CardGridTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CardGridTemplate>

export default meta
type Story = StoryObj<typeof meta>

// ============================================================================
// Stories - Use Cases (ProvidersCaseCard)
// ============================================================================

export const UseCasesGrid: Story = {
  args: {
    children: (
      <>
        <ProvidersCaseCard
          icon={<Gamepad2 />}
          title="Gaming PC Owners"
          subtitle="Most common provider type"
          quote="I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep."
          facts={[
            { label: 'Typical GPU:', value: 'RTX 4080–4090' },
            { label: 'Availability:', value: '16–20 h/day' },
            { label: 'Monthly:', value: '€120–180' },
          ]}
          index={0}
        />
        <ProvidersCaseCard
          icon={<Server />}
          title="Homelab Enthusiasts"
          subtitle="Multiple GPUs, high earnings"
          quote="Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit."
          facts={[
            { label: 'Setup:', value: '3–6 GPUs' },
            { label: 'Availability:', value: '20–24 h/day' },
            { label: 'Monthly:', value: '€300–600' },
          ]}
          index={1}
        />
        <ProvidersCaseCard
          icon={<Cpu />}
          title="Former Crypto Miners"
          subtitle="Repurpose mining rigs"
          quote="After PoS, my rig idled. rbee now earns more than mining—with better margins."
          facts={[
            { label: 'Setup:', value: '6–12 GPUs' },
            { label: 'Availability:', value: '24 h/day' },
            { label: 'Monthly:', value: '€600–1,200' },
          ]}
          index={2}
        />
        <ProvidersCaseCard
          icon={<Monitor />}
          title="Workstation Owners"
          subtitle="Professional GPUs earning"
          quote="My RTX 4080 is busy on renders only. The rest of the time it makes ~€100/mo on rbee."
          facts={[
            { label: 'Typical GPU:', value: 'RTX 4070–4080' },
            { label: 'Availability:', value: '12–16 h/day' },
            { label: 'Monthly:', value: '€80–140' },
          ]}
          index={3}
        />
      </>
    ),
  },
}

// ============================================================================
// Stories - Security (ProvidersSecurityCard)
// ============================================================================

export const SecurityGrid: Story = {
  args: {
    children: (
      <>
        <ProvidersSecurityCard
          icon={<Shield className="size-6" />}
          title="Sandboxed Execution"
          subtitle="Complete isolation"
          body="All jobs run in isolated sandboxes with no access to your files, network, or personal data."
          points={['No file system access', 'No network access', 'No personal data access', 'Automatic cleanup']}
          index={0}
        />
        <ProvidersSecurityCard
          icon={<Lock className="size-6" />}
          title="Encrypted Communication"
          subtitle="End-to-end encryption"
          body="All communication between your GPU and the marketplace is encrypted using industry-standard protocols."
          points={['TLS 1.3', 'Secure payment processing', 'Protected earnings data', 'Private job details']}
          index={1}
        />
        <ProvidersSecurityCard
          icon={<Eye className="size-6" />}
          title="Malware Scanning"
          subtitle="Automatic protection"
          body="Every job is automatically scanned for malware before execution. Suspicious jobs are blocked."
          points={['Real-time detection', 'Automatic blocking', 'Threat intel updates', 'Customer vetting']}
          index={2}
        />
        <ProvidersSecurityCard
          icon={<FileCheck className="size-6" />}
          title="Hardware Protection"
          subtitle="Warranty-safe operation"
          body="Temperature monitoring, cooldown periods, and power limits protect your hardware and warranty."
          points={['Temperature monitoring', 'Cooldown periods', 'Power limits', 'Health monitoring']}
          index={3}
        />
      </>
    ),
  },
}

// ============================================================================
// Stories - Mixed Content
// ============================================================================

export const MixedCards: Story = {
  args: {
    children: (
      <>
        <ProvidersCaseCard
          icon={<Gamepad2 />}
          title="Gaming PC Owners"
          subtitle="Most common provider type"
          quote="I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I'm at work or asleep."
          facts={[
            { label: 'Typical GPU:', value: 'RTX 4080–4090' },
            { label: 'Availability:', value: '16–20 h/day' },
            { label: 'Monthly:', value: '€120–180' },
          ]}
          index={0}
        />
        <ProvidersSecurityCard
          icon={<Shield className="size-6" />}
          title="Sandboxed Execution"
          subtitle="Complete isolation"
          body="All jobs run in isolated sandboxes with no access to your files, network, or personal data."
          points={['No file system access', 'No network access', 'No personal data access', 'Automatic cleanup']}
          index={1}
        />
      </>
    ),
  },
}

// ============================================================================
// Stories - With Custom ClassName
// ============================================================================

export const WithCustomClassName: Story = {
  args: {
    className: 'bg-secondary/50 p-8 rounded-lg',
    children: (
      <>
        <ProvidersCaseCard
          icon={<Server />}
          title="Homelab Enthusiasts"
          subtitle="Multiple GPUs, high earnings"
          quote="Four GPUs across my homelab bring ~€400/mo. It covers power and leaves profit."
          facts={[
            { label: 'Setup:', value: '3–6 GPUs' },
            { label: 'Availability:', value: '20–24 h/day' },
            { label: 'Monthly:', value: '€300–600' },
          ]}
          index={0}
        />
        <ProvidersSecurityCard
          icon={<Lock className="size-6" />}
          title="Encrypted Communication"
          subtitle="End-to-end encryption"
          body="All communication between your GPU and the marketplace is encrypted using industry-standard protocols."
          points={['TLS 1.3', 'Secure payment processing', 'Protected earnings data', 'Private job details']}
          index={1}
        />
      </>
    ),
  },
}
