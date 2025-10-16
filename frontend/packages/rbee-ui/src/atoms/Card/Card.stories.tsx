// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Globe, Shield } from 'lucide-react'
import { IconCardHeader } from '../../molecules/IconCardHeader/IconCardHeader'
import { Button } from '../Button/Button'
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './Card'

const meta: Meta<typeof Card> = {
  title: 'Atoms/Card',
  component: Card,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Card>

/**
 * ## Overview
 * Card is a flexible container component with optional header, content, and footer sections.
 * Uses CSS Grid for automatic layout of header elements including action buttons.
 *
 * ## Composition
 * - Card (container)
 * - CardHeader (title area with optional action)
 * - CardTitle (heading)
 * - CardDescription (subtitle)
 * - CardAction (header action button)
 * - CardContent (main content)
 * - CardFooter (bottom actions)
 *
 * ## When to Use
 * - Group related content
 * - Display features or benefits
 * - Show pricing tiers
 * - Present model information
 * - Create dashboard widgets
 *
 * ## Used In
 * - PricingTier
 * - FeatureCard
 * - ModelCard
 * - DashboardWidget
 * - And 10+ other organisms
 */

export const Default: Story = {
  render: () => (
    <Card className="w-[350px]">
      <CardHeader>
        <CardTitle>Card Title</CardTitle>
        <CardDescription>Card description goes here</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm">This is the main content area of the card.</p>
      </CardContent>
    </Card>
  ),
}

export const WithHeader: Story = {
  render: () => (
    <Card className="w-[350px]">
      <CardHeader>
        <CardTitle>Professional Plan</CardTitle>
        <CardDescription>Perfect for growing teams</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-sm">Price</span>
            <span className="font-semibold">€99/month</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm">Users</span>
            <span className="font-semibold">Up to 10</span>
          </div>
        </div>
      </CardContent>
    </Card>
  ),
}

export const WithFooter: Story = {
  render: () => (
    <Card className="w-[350px]">
      <CardHeader>
        <CardTitle>Get Started</CardTitle>
        <CardDescription>Deploy your first model in minutes</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          Choose from our curated collection of open-source LLMs and start serving requests immediately.
        </p>
      </CardContent>
      <CardFooter className="gap-2">
        <Button variant="outline" className="flex-1">
          Learn More
        </Button>
        <Button className="flex-1">Start Free Trial</Button>
      </CardFooter>
    </Card>
  ),
}

export const InPricingContext: Story = {
  render: () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl">
      <Card>
        <CardHeader>
          <CardTitle>Starter</CardTitle>
          <CardDescription>For individuals and small projects</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="text-3xl font-bold">€29</div>
            <div className="text-sm text-muted-foreground">per month</div>
          </div>
          <ul className="space-y-2 text-sm">
            <li>✓ 1 GPU instance</li>
            <li>✓ 100GB storage</li>
            <li>✓ Community support</li>
          </ul>
        </CardContent>
        <CardFooter>
          <Button variant="outline" className="w-full">
            Get Started
          </Button>
        </CardFooter>
      </Card>

      <Card className="border-primary">
        <CardHeader>
          <CardTitle>Professional</CardTitle>
          <CardDescription>For growing teams</CardDescription>
          <CardAction>
            <span className="text-xs font-medium text-primary">Popular</span>
          </CardAction>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="text-3xl font-bold">€99</div>
            <div className="text-sm text-muted-foreground">per month</div>
          </div>
          <ul className="space-y-2 text-sm">
            <li>✓ 5 GPU instances</li>
            <li>✓ 500GB storage</li>
            <li>✓ Priority support</li>
            <li>✓ Advanced monitoring</li>
          </ul>
        </CardContent>
        <CardFooter>
          <Button className="w-full">Start Free Trial</Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Enterprise</CardTitle>
          <CardDescription>For large organizations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="text-3xl font-bold">Custom</div>
            <div className="text-sm text-muted-foreground">contact sales</div>
          </div>
          <ul className="space-y-2 text-sm">
            <li>✓ Unlimited instances</li>
            <li>✓ Unlimited storage</li>
            <li>✓ 24/7 dedicated support</li>
            <li>✓ Custom SLA</li>
          </ul>
        </CardContent>
        <CardFooter>
          <Button variant="outline" className="w-full">
            Contact Sales
          </Button>
        </CardFooter>
      </Card>
    </div>
  ),
}

export const WithIconHeader: Story = {
  render: () => (
    <div className="grid max-w-4xl gap-6 md:grid-cols-2">
      <Card className="rounded-2xl border-border bg-card/60 p-8">
        <IconCardHeader icon={<Globe className="size-6" />} title="GDPR" subtitle="EU Regulation" titleId="card-gdpr" />
        <CardContent className="p-0">
          <p className="text-sm text-foreground/85">
            Built from the ground up to meet GDPR requirements with data processing agreements, right to erasure, and
            privacy by design.
          </p>
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-border bg-card/60 p-8">
        <IconCardHeader icon={<Shield className="size-6" />} title="SOC2" subtitle="US Standard" titleId="card-soc2" />
        <CardContent className="p-0">
          <p className="text-sm text-foreground/85">
            Security and availability controls with auditor query API, tamper-evident hash chains, and encryption at
            rest.
          </p>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        story:
          'Cards can use IconCardHeader molecule for headers with icons. This pattern is commonly used in compliance cards, feature cards, and status cards.',
      },
    },
  },
}
