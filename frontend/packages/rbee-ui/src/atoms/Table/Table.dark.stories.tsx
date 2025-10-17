// Dark Mode Showcase for Table component
import type { Meta, StoryObj } from '@storybook/react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './Table'

const meta: Meta<typeof Table> = {
  title: 'Atoms/Table/Dark Mode',
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
 * ## Dark Mode Table Showcase
 * 
 * Demonstrates neutral hover states (cool white overlays, not amber) for data density contexts:
 * - Header: bg-[rgba(255,255,255,0.03)] with text-slate-200
 * - Row hover: bg-[rgba(255,255,255,0.025)]
 * - Selected row: bg-[rgba(255,255,255,0.04)] + focus ring
 * - Numeric cells: text-slate-200 tabular-nums
 */

const mockData = [
  { model: 'Llama 3.1 70B', provider: 'RunPod', vram: '80 GB', price: '$1.89/hr', tps: '42' },
  { model: 'Mistral 7B', provider: 'Vast.ai', vram: '24 GB', price: '$0.32/hr', tps: '128' },
  { model: 'Mixtral 8x7B', provider: 'Lambda Labs', vram: '48 GB', price: '$0.89/hr', tps: '76' },
  { model: 'GPT-J 6B', provider: 'Paperspace', vram: '16 GB', price: '$0.21/hr', tps: '156' },
  { model: 'Falcon 40B', provider: 'Jarvis Labs', vram: '80 GB', price: '$1.45/hr', tps: '54' },
]

export const NeutralHoverStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">GPU Rental Pricing</h3>
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
            {mockData.map((row, i) => (
              <TableRow key={i}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>{row.provider}</TableCell>
                <TableCell>{row.vram}</TableCell>
                <TableCell className="text-right">{row.price}</TableCell>
                <TableCell className="text-right">{row.tps}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Hover over rows to see neutral white overlay (rgba(255,255,255,0.025)), not amber. Numeric cells use tabular-nums for scanning stability.',
      },
    },
  },
}

export const SelectedRowWithFocus: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Selected State</h3>
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
            <TableRow>
              <TableCell className="font-medium">{mockData[0].model}</TableCell>
              <TableCell>{mockData[0].provider}</TableCell>
              <TableCell>{mockData[0].vram}</TableCell>
              <TableCell className="text-right">{mockData[0].price}</TableCell>
              <TableCell className="text-right">{mockData[0].tps}</TableCell>
            </TableRow>
            <TableRow data-state="selected" tabIndex={0}>
              <TableCell className="font-medium">{mockData[1].model}</TableCell>
              <TableCell>{mockData[1].provider}</TableCell>
              <TableCell>{mockData[1].vram}</TableCell>
              <TableCell className="text-right">{mockData[1].price}</TableCell>
              <TableCell className="text-right">{mockData[1].tps}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="font-medium">{mockData[2].model}</TableCell>
              <TableCell>{mockData[2].provider}</TableCell>
              <TableCell>{mockData[2].vram}</TableCell>
              <TableCell className="text-right">{mockData[2].price}</TableCell>
              <TableCell className="text-right">{mockData[2].tps}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Selected row (middle) shows bg-[rgba(255,255,255,0.04)] with amber focus ring for keyboard navigation parity.',
      },
    },
  },
}

export const HeaderAndFooter: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg">
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">Full Table Structure</h3>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead className="text-right">Price</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {mockData.slice(0, 3).map((row, i) => (
              <TableRow key={i}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>{row.provider}</TableCell>
                <TableCell className="text-right">{row.price}</TableCell>
              </TableRow>
            ))}
          </TableBody>
          <tfoot className="bg-[rgba(255,255,255,0.03)] border-t font-medium">
            <tr>
              <td className="p-2" colSpan={2}>
                Total
              </td>
              <td className="p-2 text-right tabular-nums">$4.10/hr</td>
            </tr>
          </tfoot>
        </Table>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Header and footer use bg-[rgba(255,255,255,0.03)] for subtle separation. Text is text-slate-200 for legibility.',
      },
    },
  },
}
