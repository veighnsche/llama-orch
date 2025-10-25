import { Tabs, TabsList } from '@rbee/ui/atoms/Tabs'
import type { Meta, StoryObj } from '@storybook/react'
import { Code, Cpu, Gauge, Sparkles, Zap } from 'lucide-react'
import { FeatureTab } from './FeatureTab'

const meta: Meta<typeof FeatureTab> = {
  title: 'Molecules/FeatureTab',
  component: FeatureTab,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <Tabs defaultValue="api" className="w-full max-w-2xl">
        <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto rounded-md border bg-card/60 p-1 gap-1">
          <Story />
        </TabsList>
      </Tabs>
    ),
  ],
}

export default meta
type Story = StoryObj<typeof meta>

export const API: Story = {
  args: {
    value: 'api',
    icon: <Code className="size-6" />,
    label: 'OpenAI-Compatible',
    mobileLabel: 'OpenAI',
  },
}

export const GPU: Story = {
  args: {
    value: 'gpu',
    icon: <Cpu className="size-6" />,
    label: 'Multi-GPU',
    mobileLabel: 'GPU',
  },
}

export const Scheduler: Story = {
  args: {
    value: 'scheduler',
    icon: <Gauge className="size-6" />,
    label: 'Scheduler',
    mobileLabel: 'Rhai',
  },
}

export const RealTime: Story = {
  args: {
    value: 'sse',
    icon: <Zap className="size-6" />,
    label: 'Real‑time',
    mobileLabel: 'SSE',
  },
}

export const WithoutMobileLabel: Story = {
  args: {
    value: 'feature',
    icon: <Sparkles className="size-6" />,
    label: 'Feature Name',
  },
}

export const AllTabs: Story = {
  args: {
    value: 'api',
    icon: <Code className="size-6" />,
    label: 'OpenAI-Compatible',
  },
  decorators: [
    () => (
      <Tabs defaultValue="api" className="w-full max-w-4xl">
        <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto rounded-md border bg-card/60 p-1 gap-1">
          <FeatureTab value="api" icon={<Code className="size-6" />} label="OpenAI-Compatible" mobileLabel="OpenAI" />
          <FeatureTab value="gpu" icon={<Cpu className="size-6" />} label="Multi-GPU" mobileLabel="GPU" />
          <FeatureTab value="scheduler" icon={<Gauge className="size-6" />} label="Scheduler" mobileLabel="Rhai" />
          <FeatureTab value="sse" icon={<Zap className="size-6" />} label="Real‑time" mobileLabel="SSE" />
        </TabsList>
      </Tabs>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story: 'Example showing all feature tabs together in a responsive grid.',
      },
    },
  },
}
