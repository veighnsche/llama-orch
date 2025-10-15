// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './Tabs'

const meta: Meta<typeof Tabs> = {
  title: 'Atoms/Tabs',
  component: Tabs,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Tabs>

/**
 * ## Overview
 * Tabs organize content into multiple panels, showing one panel at a time.
 * Built on Radix UI Tabs primitive with responsive horizontal/vertical layouts.
 *
 * ## Composition
 * - Tabs (container)
 * - TabsList (tab buttons container)
 * - TabsTrigger (individual tab button)
 * - TabsContent (panel content)
 *
 * ## When to Use
 * - Switch between related views
 * - Organize complex interfaces
 * - Show different data perspectives
 * - Group related settings
 *
 * ## Used In
 * - FeaturesSection (feature categories)
 * - PricingSection (billing periods)
 * - DashboardView (data views)
 */

export const Default: Story = {
  render: () => (
    <Tabs defaultValue="tab1" className="w-[400px]">
      <TabsList>
        <TabsTrigger value="tab1">Account</TabsTrigger>
        <TabsTrigger value="tab2">Password</TabsTrigger>
        <TabsTrigger value="tab3">Notifications</TabsTrigger>
      </TabsList>
      <TabsContent value="tab1">
        <div className="p-4 border rounded-lg">
          <h3 className="font-semibold mb-2">Account Settings</h3>
          <p className="text-sm text-muted-foreground">Manage your account preferences and profile information.</p>
        </div>
      </TabsContent>
      <TabsContent value="tab2">
        <div className="p-4 border rounded-lg">
          <h3 className="font-semibold mb-2">Password Settings</h3>
          <p className="text-sm text-muted-foreground">Update your password and security settings.</p>
        </div>
      </TabsContent>
      <TabsContent value="tab3">
        <div className="p-4 border rounded-lg">
          <h3 className="font-semibold mb-2">Notification Preferences</h3>
          <p className="text-sm text-muted-foreground">Choose how you want to be notified.</p>
        </div>
      </TabsContent>
    </Tabs>
  ),
}

export const WithIcons: Story = {
  render: () => (
    <Tabs defaultValue="models" className="w-[400px]">
      <TabsList>
        <TabsTrigger value="models">
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          Models
        </TabsTrigger>
        <TabsTrigger value="deployments">
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Deployments
        </TabsTrigger>
        <TabsTrigger value="analytics">
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          Analytics
        </TabsTrigger>
      </TabsList>
      <TabsContent value="models">
        <div className="p-4 border rounded-lg">
          <p className="text-sm">Browse and deploy AI models</p>
        </div>
      </TabsContent>
      <TabsContent value="deployments">
        <div className="p-4 border rounded-lg">
          <p className="text-sm">Manage your active deployments</p>
        </div>
      </TabsContent>
      <TabsContent value="analytics">
        <div className="p-4 border rounded-lg">
          <p className="text-sm">View usage statistics and metrics</p>
        </div>
      </TabsContent>
    </Tabs>
  ),
}

export const Vertical: Story = {
  render: () => (
    <Tabs defaultValue="overview" className="w-[600px]">
      <div className="flex gap-4">
        <TabsList className="flex-col items-stretch">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="billing">Billing</TabsTrigger>
        </TabsList>
        <div className="flex-1">
          <TabsContent value="overview">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Overview</h3>
              <p className="text-sm text-muted-foreground">Dashboard overview and key metrics</p>
            </div>
          </TabsContent>
          <TabsContent value="performance">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Performance</h3>
              <p className="text-sm text-muted-foreground">Monitor system performance and uptime</p>
            </div>
          </TabsContent>
          <TabsContent value="security">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Security</h3>
              <p className="text-sm text-muted-foreground">Security settings and access control</p>
            </div>
          </TabsContent>
          <TabsContent value="billing">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">Billing</h3>
              <p className="text-sm text-muted-foreground">Invoices and payment methods</p>
            </div>
          </TabsContent>
        </div>
      </div>
    </Tabs>
  ),
}

export const InFeaturesContext: Story = {
  render: () => (
    <div className="max-w-4xl">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2">Platform Features</h2>
        <p className="text-muted-foreground">Explore what makes our platform powerful</p>
      </div>
      <Tabs defaultValue="deployment" className="w-full">
        <TabsList className="w-full justify-center">
          <TabsTrigger value="deployment">Fast Deployment</TabsTrigger>
          <TabsTrigger value="scaling">Auto Scaling</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
        </TabsList>
        <TabsContent value="deployment" className="mt-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Deploy in Seconds</h3>
              <p className="text-muted-foreground mb-4">
                Get your models running in production with just a few clicks. No complex configuration required.
              </p>
              <ul className="space-y-2 text-sm">
                <li>✓ One-click deployment</li>
                <li>✓ Pre-configured environments</li>
                <li>✓ Automatic dependency management</li>
              </ul>
            </div>
            <div className="bg-muted rounded-lg p-6 flex items-center justify-center">
              <div className="text-center text-muted-foreground">Deployment Demo</div>
            </div>
          </div>
        </TabsContent>
        <TabsContent value="scaling" className="mt-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Scale Automatically</h3>
              <p className="text-muted-foreground mb-4">
                Handle traffic spikes effortlessly with intelligent auto-scaling based on demand.
              </p>
              <ul className="space-y-2 text-sm">
                <li>✓ Real-time load balancing</li>
                <li>✓ Cost-optimized scaling</li>
                <li>✓ Zero-downtime updates</li>
              </ul>
            </div>
            <div className="bg-muted rounded-lg p-6 flex items-center justify-center">
              <div className="text-center text-muted-foreground">Scaling Demo</div>
            </div>
          </div>
        </TabsContent>
        <TabsContent value="monitoring" className="mt-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Monitor Everything</h3>
              <p className="text-muted-foreground mb-4">
                Get real-time insights into your deployments with comprehensive monitoring and alerting.
              </p>
              <ul className="space-y-2 text-sm">
                <li>✓ Performance metrics</li>
                <li>✓ Custom alerts</li>
                <li>✓ Detailed logs</li>
              </ul>
            </div>
            <div className="bg-muted rounded-lg p-6 flex items-center justify-center">
              <div className="text-center text-muted-foreground">Monitoring Demo</div>
            </div>
          </div>
        </TabsContent>
        <TabsContent value="security" className="mt-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Enterprise Security</h3>
              <p className="text-muted-foreground mb-4">
                Keep your data safe with industry-leading security practices and compliance certifications.
              </p>
              <ul className="space-y-2 text-sm">
                <li>✓ End-to-end encryption</li>
                <li>✓ SOC 2 compliant</li>
                <li>✓ Role-based access control</li>
              </ul>
            </div>
            <div className="bg-muted rounded-lg p-6 flex items-center justify-center">
              <div className="text-center text-muted-foreground">Security Demo</div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  ),
}
