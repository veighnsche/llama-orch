import type { Meta, StoryObj } from '@storybook/react'
import { Disclaimer } from './Disclaimer'

const meta = {
  title: 'Molecules/Disclaimer',
  component: Disclaimer,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof Disclaimer>

export default meta
type Story = StoryObj<typeof meta>

const sampleText =
  'Earnings are estimates based on current market rates and may vary. Actual earnings depend on GPU utilization, demand, and network conditions. Past performance does not guarantee future results.'

export const Default: Story = {
  args: {
    children: sampleText,
    variant: 'default',
  },
}

export const WithIcon: Story = {
  args: {
    children: sampleText,
    variant: 'default',
    showIcon: true,
  },
}

export const Info: Story = {
  args: {
    children: 'This feature is available on all paid plans. Contact sales for enterprise pricing.',
    variant: 'info',
    showIcon: true,
  },
}

export const Warning: Story = {
  args: {
    children: 'This feature is currently in beta. Some functionality may change before general availability.',
    variant: 'warning',
    showIcon: true,
  },
}

export const Muted: Story = {
  args: {
    children: 'All prices shown are in EUR and exclude VAT. Prices subject to change.',
    variant: 'muted',
  },
}

export const LongText: Story = {
  args: {
    children: (
      <>
        <strong>Important Legal Notice:</strong> The information provided on this platform is for general informational
        purposes only. All earnings calculations are estimates based on current market conditions and historical data.
        Actual results may vary significantly based on factors including but not limited to: GPU model, availability,
        market demand, network conditions, and operational costs. By using this service, you acknowledge that past
        performance is not indicative of future results and that rbee makes no guarantees regarding earnings potential.
      </>
    ),
    variant: 'default',
    showIcon: true,
  },
}

export const MultipleVariants: Story = {
  args: {
    children: '',
  },
  render: () => (
    <div className="space-y-4 max-w-2xl">
      <Disclaimer variant="default" showIcon>
        Default variant with icon - suitable for general disclaimers and legal notices.
      </Disclaimer>
      <Disclaimer variant="info" showIcon>
        Info variant - use for helpful information and tips.
      </Disclaimer>
      <Disclaimer variant="warning" showIcon>
        Warning variant - use for important notices that need attention.
      </Disclaimer>
      <Disclaimer variant="muted">Muted variant - use for less critical supplementary information.</Disclaimer>
    </div>
  ),
}
