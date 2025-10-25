// Dark Mode Showcase for Card component
import type { Meta, StoryObj } from '@storybook/react'
import { Database, Globe, Server, Shield } from 'lucide-react'
import { IconCardHeader } from '../../molecules/IconCardHeader/IconCardHeader'
import { Button } from '../Button/Button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './Card'

const meta: Meta<typeof Card> = {
  title: 'Atoms/Card/Dark Mode',
  component: Card,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#0b1220' }],
    },
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <Story />
      </div>
    ),
  ],
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Card>

/**
 * ## Dark Mode Card Showcase
 *
 * Demonstrates the refined dark theme with:
 * - Rich surfaces (#141c2a cards on #0b1220 canvas)
 * - Reduced glare (#e5eaf1 foreground)
 * - Controlled amber presence
 * - Dark-friendly elevation shadows with ambient + highlight insets
 */

export const CardOnCanvas: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <Card className="w-[350px]">
        <CardHeader>
          <CardTitle>Dark Mode Card</CardTitle>
          <CardDescription>Rich surface with clear hierarchy</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-foreground/85">
            Canvas (#0b1220) → Card (#141c2a) creates ~8% lift for readable depth without harsh contrast.
          </p>
        </CardContent>
        <CardFooter className="gap-2">
          <Button variant="outline" className="flex-1">
            Secondary
          </Button>
          <Button className="flex-1">Primary CTA</Button>
        </CardFooter>
      </Card>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Card on canvas background showing the refined dark surface hierarchy. Note the subtle shadow with inset highlight.',
      },
    },
  },
}

export const PopoverOnCard: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <Card className="w-[400px]">
        <CardHeader>
          <CardTitle>Card Surface</CardTitle>
          <CardDescription>Base layer (#141c2a)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-popover border border-border rounded-md p-4 [box-shadow:var(--shadow-md)]">
            <h4 className="font-semibold mb-2">Popover Layer</h4>
            <p className="text-sm text-muted-foreground">
              Popover (#161f2e) sits above card with tiny lift for overlay separation.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Demonstrates surface stacking: Canvas → Card → Popover with clear visual hierarchy.',
      },
    },
  },
}

export const FormOnCard: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <Card className="w-[400px]">
        <CardHeader>
          <CardTitle>Sign In</CardTitle>
          <CardDescription>Enter your credentials</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="email" className="text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              placeholder="you@example.com"
              className="file:text-foreground placeholder:text-[#8b9bb0] selection:bg-primary selection:text-primary-foreground bg-[#0f172a] border-input h-9 w-full min-w-0 rounded-md border px-3 py-1 text-base transition-[color,box-shadow,border-color] [box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)] hover:border-slate-400 focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background md:text-sm"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="password" className="text-sm font-medium">
              Password
            </label>
            <input
              id="password"
              type="password"
              placeholder="••••••••"
              className="file:text-foreground placeholder:text-[#8b9bb0] selection:bg-primary selection:text-primary-foreground bg-[#0f172a] border-input h-9 w-full min-w-0 rounded-md border px-3 py-1 text-base transition-[color,box-shadow,border-color] [box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)] hover:border-slate-400 focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background md:text-sm"
            />
          </div>
        </CardContent>
        <CardFooter>
          <Button className="w-full">Sign In</Button>
        </CardFooter>
      </Card>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Form inputs with dark mode inset shadows and improved placeholder legibility (#8b9bb0).',
      },
    },
  },
}

export const IconHeaderCards: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <div className="grid max-w-4xl gap-6 md:grid-cols-2">
        <Card className="rounded border-border bg-card/60 p-8">
          <IconCardHeader
            icon={<Globe className="size-6" />}
            title="GDPR"
            subtitle="EU Regulation"
            titleId="card-gdpr-dark"
          />
          <CardContent className="p-0">
            <p className="text-sm text-foreground/85">
              Built from the ground up to meet GDPR requirements with data processing agreements, right to erasure, and
              privacy by design.
            </p>
          </CardContent>
        </Card>

        <Card className="rounded border-border bg-card/60 p-8">
          <IconCardHeader
            icon={<Shield className="size-6" />}
            title="SOC2"
            subtitle="US Standard"
            titleId="card-soc2-dark"
          />
          <CardContent className="p-0">
            <p className="text-sm text-foreground/85">
              Security and availability controls with auditor query API, tamper-evident hash chains, and encryption at
              rest.
            </p>
          </CardContent>
        </Card>

        <Card className="rounded border-border bg-card/60 p-8">
          <IconCardHeader
            icon={<Server className="size-6" />}
            title="ISO 27001"
            subtitle="Security"
            titleId="card-iso-dark"
          />
          <CardContent className="p-0">
            <p className="text-sm text-foreground/85">
              Information security management system with risk assessment, access controls, and continuous monitoring.
            </p>
          </CardContent>
        </Card>

        <Card className="rounded border-border bg-card/60 p-8">
          <IconCardHeader
            icon={<Database className="size-6" />}
            title="HIPAA"
            subtitle="Healthcare"
            titleId="card-hipaa-dark"
          />
          <CardContent className="p-0">
            <p className="text-sm text-foreground/85">
              Protected health information safeguards with encryption, audit logs, and business associate agreements.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Icon header cards showing consistent spacing and amber restraint (icons only, no amber backgrounds).',
      },
    },
  },
}

export const BrandProgression: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded">
      <Card className="w-[400px]">
        <CardHeader>
          <CardTitle>Brand Progression</CardTitle>
          <CardDescription>Amber authority without glare</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Primary Button States:</p>
            <div className="flex gap-2">
              <Button size="sm">Default (#b45309)</Button>
              <Button size="sm" className="bg-[#d97706]">
                Hover (#d97706)
              </Button>
              <Button size="sm" className="bg-[#92400e]">
                Active (#92400e)
              </Button>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium">Link States:</p>
            <div className="space-y-1">
              <a
                href="#"
                className="text-[color:var(--accent)] underline underline-offset-2 decoration-amber-400 text-sm"
              >
                Default link (#d97706)
              </a>
              <br />
              <a href="#" className="text-white underline underline-offset-2 decoration-amber-400 text-sm">
                Hover link (white + amber-400)
              </a>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Brand color progression: #b45309 (default) → #d97706 (hover) → #92400e (active). Links use #d97706 default, white on hover.',
      },
    },
  },
}
