import type { Meta, StoryObj } from '@storybook/react'
import { EuLedgerGrid } from './EuLedgerGrid'

const meta = {
  title: 'Atoms/EuLedgerGrid',
  component: EuLedgerGrid,
  parameters: {
    layout: 'fullscreen',
    backgrounds: {
      default: 'light',
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EuLedgerGrid>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default EuLedgerGrid as background decoration
 */
export const Default: Story = {
  render: () => (
    <div className="relative h-screen w-full bg-background">
      <EuLedgerGrid className="absolute inset-0 opacity-15" />
      <div className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-2xl p-8 text-center">
          <h1 className="mb-4 text-4xl font-bold text-foreground">EU-Compliant AI Infrastructure</h1>
          <p className="text-lg text-muted-foreground">
            Abstract ledger grid with glowing checkpoints, implying immutable audit trails and data sovereignty.
          </p>
        </div>
      </div>
    </div>
  ),
}

/**
 * Light theme variant
 */
export const LightTheme: Story = {
  parameters: {
    backgrounds: {
      default: 'light',
    },
  },
  render: () => (
    <div className="relative h-screen w-full bg-white">
      <EuLedgerGrid className="absolute inset-0 opacity-20" />
      <div className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-2xl rounded-xl border bg-card p-8 text-center shadow-lg">
          <h2 className="mb-2 text-2xl font-bold text-foreground">Light Theme</h2>
          <p className="text-muted-foreground">Subtle blue grid with low opacity for light backgrounds</p>
        </div>
      </div>
    </div>
  ),
}

/**
 * Dark theme variant
 */
export const DarkTheme: Story = {
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  render: () => (
    <div className="dark relative h-screen w-full bg-slate-950">
      <EuLedgerGrid className="absolute inset-0 opacity-20" />
      <div className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-2xl rounded-xl border border-slate-800 bg-slate-900 p-8 text-center shadow-lg">
          <h2 className="mb-2 text-2xl font-bold text-white">Dark Theme</h2>
          <p className="text-slate-400">Brighter blue grid with glow effect for dark backgrounds</p>
        </div>
      </div>
    </div>
  ),
}

/**
 * As used in Enterprise Solution section
 */
export const InEnterpriseSolution: Story = {
  render: () => (
    <div className="relative min-h-screen w-full bg-background">
      <EuLedgerGrid className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 md:block" />
      <div className="relative z-10 mx-auto max-w-7xl px-6 py-24">
        <div className="text-center">
          <h2 className="mb-4 text-3xl font-bold text-foreground">How rbee Works</h2>
          <p className="mb-8 text-lg text-muted-foreground">
            GDPR-compliant by design. SOC2 ready. ISO 27001 aligned.
          </p>
          <div className="grid gap-6 md:grid-cols-3">
            {['Deploy On-Premises', 'Enable Audit Logging', 'Run Compliant AI'].map((title, i) => (
              <div key={i} className="rounded-lg border bg-card p-6">
                <h3 className="mb-2 font-semibold text-foreground">{title}</h3>
                <p className="text-sm text-muted-foreground">
                  Your models, your data, your infrastructure. Zero external dependencies.
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  ),
}

/**
 * High opacity variant
 */
export const HighOpacity: Story = {
  render: () => (
    <div className="relative h-screen w-full bg-background">
      <EuLedgerGrid className="absolute inset-0 opacity-40" />
      <div className="relative z-10 flex h-full items-center justify-center">
        <div className="max-w-2xl p-8 text-center">
          <h2 className="mb-2 text-2xl font-bold text-foreground">Higher Opacity</h2>
          <p className="text-muted-foreground">More visible grid pattern for emphasis</p>
        </div>
      </div>
    </div>
  ),
}

/**
 * Positioned variant (top-right)
 */
export const PositionedTopRight: Story = {
  render: () => (
    <div className="relative h-screen w-full bg-background">
      <EuLedgerGrid className="pointer-events-none absolute -right-64 -top-32 w-[60rem] opacity-10" />
      <div className="relative z-10 flex h-full items-start justify-start p-12">
        <div className="max-w-xl">
          <h2 className="mb-4 text-3xl font-bold text-foreground">Positioned Decoration</h2>
          <p className="text-lg text-muted-foreground">
            SVG can be positioned anywhere as a decorative background element
          </p>
        </div>
      </div>
    </div>
  ),
}
