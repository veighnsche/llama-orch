import type { Meta, StoryObj } from '@storybook/react'
import * as Icons from './index'

const meta = {
  title: 'Icons/All Icons',
  parameters: {
    layout: 'padded',
  },
} satisfies Meta

export default meta

type Story = StoryObj<typeof meta>

export const AllIcons: Story = {
  render: () => {
    const iconEntries = Object.entries(Icons)
    
    return (
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold mb-4">All Icons ({iconEntries.length})</h2>
          <p className="text-muted-foreground mb-6">
            All SVG illustrations converted to React components with TypeScript support.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
          {iconEntries
            .filter(([name]) => name !== 'HoneycombPattern' && name !== 'BeeGlyph')
            .map(([name, Component]) => {
              // Type assertion: filtered components all have numeric size prop
              const IconComponent = Component as React.ComponentType<{ size?: number | string; className?: string }>
              return (
                <div
                  key={name}
                  className="flex flex-col items-center gap-3 p-4 rounded-lg border hover:bg-accent transition-colors"
                >
                  <IconComponent size={48} className="text-foreground" />
                  <span className="text-xs text-center font-mono text-muted-foreground">
                    {name}
                  </span>
                </div>
              )
            })}
        </div>

        <div className="mt-12 p-6 rounded-lg bg-muted">
          <h3 className="text-lg font-semibold mb-3">Usage</h3>
          <pre className="text-sm bg-background p-4 rounded overflow-x-auto">
            <code>{`import { BeeMark, BeeGlyph } from '@rbee/ui/icons'

<BeeMark size={32} className="text-primary" />
<BeeGlyph size={48} />
`}</code>
          </pre>
        </div>
      </div>
    )
  },
}

export const Sizes: Story = {
  render: () => {
    const sizes = [16, 24, 32, 48, 64, 96]
    
    return (
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold mb-2">Icon Sizes</h2>
          <p className="text-muted-foreground">
            All icons support custom sizes via the <code>size</code> prop.
          </p>
        </div>

        <div className="flex items-end gap-8 flex-wrap">
          {sizes.map((size) => (
            <div key={size} className="flex flex-col items-center gap-2">
              <Icons.BeeMark size={size} className="text-primary" />
              <span className="text-xs text-muted-foreground">{size}px</span>
            </div>
          ))}
        </div>
      </div>
    )
  },
}

export const Colors: Story = {
  render: () => {
    const colorClasses = [
      { name: 'Primary', class: 'text-primary' },
      { name: 'Secondary', class: 'text-secondary-foreground' },
      { name: 'Muted', class: 'text-muted-foreground' },
      { name: 'Amber 500', class: 'text-amber-500' },
      { name: 'Blue 600', class: 'text-blue-600' },
      { name: 'Red 500', class: 'text-red-500' },
    ]
    
    return (
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold mb-2">Icon Colors</h2>
          <p className="text-muted-foreground">
            Icons inherit color from <code>currentColor</code> or can be styled via <code>className</code>.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-6">
          {colorClasses.map(({ name, class: className }) => (
            <div key={name} className="flex flex-col items-center gap-3">
              <Icons.BeeMark size={48} className={className} />
              <span className="text-xs text-center text-muted-foreground">{name}</span>
            </div>
          ))}
        </div>
      </div>
    )
  },
}

export const BrandIcons: Story = {
  render: () => {
    return (
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold mb-2">Brand Icons</h2>
          <p className="text-muted-foreground">Primary brand identity icons.</p>
        </div>

        <div className="flex gap-12 items-center flex-wrap">
          <div className="flex flex-col items-center gap-3">
            <Icons.BeeMark size={64} className="text-amber-500" />
            <span className="text-sm font-medium">BeeMark</span>
          </div>
          <div className="flex flex-col items-center gap-3">
            <Icons.BeeGlyph className="text-primary" />
            <span className="text-sm font-medium">BeeGlyph</span>
          </div>
          <div className="flex flex-col items-center gap-3">
            <Icons.HomelabBee size={64} className="text-foreground" />
            <span className="text-sm font-medium">HomelabBee</span>
          </div>
        </div>
      </div>
    )
  },
}

export const SocialIcons: Story = {
  render: () => {
    return (
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold mb-2">Social Media Icons</h2>
          <p className="text-muted-foreground">Icons for social media platforms.</p>
        </div>

        <div className="flex gap-8 items-center flex-wrap">
          <div className="flex flex-col items-center gap-3">
            <Icons.GitHubIcon size={48} className="text-foreground" />
            <span className="text-sm">GitHub</span>
          </div>
          <div className="flex flex-col items-center gap-3">
            <Icons.XTwitterIcon size={48} className="text-foreground" />
            <span className="text-sm">X/Twitter</span>
          </div>
          <div className="flex flex-col items-center gap-3">
            <Icons.DiscordIcon size={48} className="text-indigo-500" />
            <span className="text-sm">Discord</span>
          </div>
        </div>
      </div>
    )
  },
}
