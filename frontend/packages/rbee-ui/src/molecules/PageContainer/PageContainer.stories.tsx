import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms'
import type { Meta, StoryObj } from '@storybook/react'
import { PageContainer } from './PageContainer'

const meta = {
  title: 'Molecules/PageContainer',
  component: PageContainer,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof PageContainer>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Dashboard',
    description: 'Monitor your queen, hives, workers, and models',
    children: (
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Queen Status</CardTitle>
            <CardDescription>Central orchestrator</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-green-500" />
              <span>Connected</span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Hives</CardTitle>
            <CardDescription>Pool managers</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">3</p>
            <p className="text-sm text-muted-foreground">2 available</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Workers</CardTitle>
            <CardDescription>Active executors</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">12</p>
            <p className="text-sm text-muted-foreground">8 available</p>
          </CardContent>
        </Card>
      </div>
    ),
  },
}

export const WithActions: Story = {
  args: {
    title: 'Settings',
    description: 'Configure your rbee installation',
    actions: (
      <>
        <Button variant="outline" size="sm">
          Reset
        </Button>
        <Button size="sm">Save Changes</Button>
      </>
    ),
    children: (
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Queen Configuration</CardTitle>
            <CardDescription>Configure queen rbee settings</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Hive Settings</CardTitle>
            <CardDescription>Manage hive configurations</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>
      </div>
    ),
  },
}

export const CompactSpacing: Story = {
  args: {
    title: 'Help & Documentation',
    description: 'Get started with rbee',
    spacing: 'compact',
    children: (
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quick Start</CardTitle>
            <CardDescription>Get up and running quickly</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Learn the basics of rbee.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>API Reference</CardTitle>
            <CardDescription>Complete API documentation</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Detailed documentation for all endpoints.</p>
          </CardContent>
        </Card>
      </div>
    ),
  },
}

export const RelaxedSpacing: Story = {
  args: {
    title: 'Models',
    description: 'Manage your downloaded models',
    spacing: 'relaxed',
    actions: <Button size="sm">Download Model</Button>,
    children: (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Llama 3.2 3B</CardTitle>
            <CardDescription>Lightweight model for quick inference</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Downloaded • 3.2 GB</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Llama 3.1 8B</CardTitle>
            <CardDescription>Balanced model for general use</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Downloaded • 8.1 GB</p>
          </CardContent>
        </Card>
      </div>
    ),
  },
}

export const NoDescription: Story = {
  args: {
    title: 'Workers',
    children: (
      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Worker 1</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Status: Active</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Worker 2</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Status: Idle</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Worker 3</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Status: Active</p>
          </CardContent>
        </Card>
      </div>
    ),
  },
}
