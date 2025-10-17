// Dark Mode Polish Showcase for Tables (Sticky Headers & Focus)
import type { Meta, StoryObj } from '@storybook/react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './Table'

const meta: Meta<typeof Table> = {
  title: 'Atoms/Table/Dark Mode Sticky',
  component: Table,
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
type Story = StoryObj<typeof Table>

/**
 * ## Dark Mode Polish: Sticky Headers & Focus
 *
 * Demonstrates:
 * - Sticky headers: backdrop-blur-[2px] + bg-[rgba(20,28,42,0.85)]
 * - Row focus: ring-[color:var(--ring)] with proper offset
 * - Keyboard navigation: Clear focus path without amber flooding
 */

const mockData = Array.from({ length: 20 }, (_, i) => ({
  id: i + 1,
  model: `Model ${i + 1}`,
  provider: ['RunPod', 'Vast.ai', 'Lambda Labs', 'Paperspace'][i % 4],
  vram: ['24 GB', '48 GB', '80 GB'][i % 3],
  price: `$${(Math.random() * 2 + 0.2).toFixed(2)}/hr`,
  tps: Math.floor(Math.random() * 100 + 30),
}))

export const StickyHeader: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Sticky Table Header</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Scroll down to see the sticky header remain visible with backdrop blur. The header uses backdrop-blur-[2px] +
          bg-[rgba(20,28,42,0.85)] to match the card/canvas mix.
        </p>

        <div className="max-h-[400px] overflow-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead sticky>ID</TableHead>
                <TableHead sticky>Model</TableHead>
                <TableHead sticky>Provider</TableHead>
                <TableHead sticky>VRAM</TableHead>
                <TableHead sticky className="text-right">
                  Price
                </TableHead>
                <TableHead sticky className="text-right">
                  TPS
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockData.map((row) => (
                <TableRow key={row.id}>
                  <TableCell className="font-medium">{row.id}</TableCell>
                  <TableCell>{row.model}</TableCell>
                  <TableCell>{row.provider}</TableCell>
                  <TableCell>{row.vram}</TableCell>
                  <TableCell className="text-right">{row.price}</TableCell>
                  <TableCell className="text-right">{row.tps}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <div className="mt-4 p-4 bg-muted rounded-md">
          <p className="text-sm font-medium mb-2">Technical Details</p>
          <ul className="list-disc list-inside text-xs space-y-1 text-muted-foreground">
            <li>sticky top-0 z-10 (stays at top while scrolling)</li>
            <li>backdrop-blur-[2px] (subtle blur for readability)</li>
            <li>bg-[rgba(20,28,42,0.85)] (matches card/canvas mix)</li>
            <li>text-slate-200 (legible header text)</li>
          </ul>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Sticky table header with backdrop blur remaining readable while scrolling.',
      },
    },
  },
}

export const KeyboardFocus: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Keyboard Navigation</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Tab through the rows to see the focus ring. Uses ring-[color:var(--ring)] with proper offset for clear
          keyboard path without amber flooding the grid.
        </p>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>VRAM</TableHead>
              <TableHead className="text-right">Price</TableHead>
              <TableHead className="text-right">TPS</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {mockData.slice(0, 5).map((row) => (
              <TableRow key={row.id} tabIndex={0}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>{row.provider}</TableCell>
                <TableCell>{row.vram}</TableCell>
                <TableCell className="text-right">{row.price}</TableCell>
                <TableCell className="text-right">{row.tps}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="mt-4 p-4 bg-muted rounded-md">
          <p className="text-sm font-medium mb-2">Focus Ring Details</p>
          <ul className="list-disc list-inside text-xs space-y-1 text-muted-foreground">
            <li>focus-visible:ring-2 (2px ring width)</li>
            <li>focus-visible:ring-[color:var(--ring)] (uses token)</li>
            <li>focus-visible:ring-offset-2 (2px offset)</li>
            <li>focus-visible:ring-offset-[color:var(--background)] (canvas color)</li>
            <li>Amber ring (#b45309) for brand consistency</li>
          </ul>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Keyboard navigation showing clear focus ring on table rows.',
      },
    },
  },
}

export const SelectedAndFocused: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Selected + Focused States</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Shows the difference between hover, selected, and focused states. All use neutral white overlays (not amber)
          for data density contexts.
        </p>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>State</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead className="text-right">Price</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell className="font-medium">Default</TableCell>
              <TableCell>{mockData[0].model}</TableCell>
              <TableCell>{mockData[0].provider}</TableCell>
              <TableCell className="text-right">{mockData[0].price}</TableCell>
            </TableRow>
            <TableRow className="hover:bg-[rgba(255,255,255,0.025)]">
              <TableCell className="font-medium">Hover</TableCell>
              <TableCell>{mockData[1].model}</TableCell>
              <TableCell>{mockData[1].provider}</TableCell>
              <TableCell className="text-right">{mockData[1].price}</TableCell>
            </TableRow>
            <TableRow data-state="selected">
              <TableCell className="font-medium">Selected</TableCell>
              <TableCell>{mockData[2].model}</TableCell>
              <TableCell>{mockData[2].provider}</TableCell>
              <TableCell className="text-right">{mockData[2].price}</TableCell>
            </TableRow>
            <TableRow
              data-state="selected"
              tabIndex={0}
              className="ring-2 ring-[color:var(--ring)] ring-offset-2 ring-offset-[color:var(--background)]"
            >
              <TableCell className="font-medium">Selected + Focused</TableCell>
              <TableCell>{mockData[3].model}</TableCell>
              <TableCell>{mockData[3].provider}</TableCell>
              <TableCell className="text-right">{mockData[3].price}</TableCell>
            </TableRow>
          </TableBody>
        </Table>

        <div className="mt-4 p-4 bg-muted rounded-md">
          <p className="text-sm font-medium mb-2">State Backgrounds</p>
          <ul className="list-disc list-inside text-xs space-y-1 text-muted-foreground">
            <li>Hover: rgba(255,255,255,0.025) — Subtle neutral</li>
            <li>Selected: rgba(255,255,255,0.04) — Slightly stronger</li>
            <li>Focused: Amber ring (#b45309) for keyboard parity</li>
            <li>All neutral (not amber) to avoid flooding data grids</li>
          </ul>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Table row states showing hover, selected, and focused with neutral backgrounds.',
      },
    },
  },
}
