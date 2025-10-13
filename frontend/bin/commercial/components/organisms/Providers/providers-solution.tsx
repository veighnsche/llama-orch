import { DollarSign, Shield, Sliders, Zap } from 'lucide-react'
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'

export function ProvidersSolution() {
  return (
    <SolutionSection
      id="how-it-works"
      kicker="How rbee Works"
      title="Turn Idle GPUs Into Reliable Monthly Income"
      subtitle="rbee connects your GPUs with developers who need compute. You set the price, control availability, and get paid automatically."
      features={[
        {
          icon: <DollarSign className="h-8 w-8" aria-hidden="true" />,
          title: 'Passive Income',
          body: 'Earn €50–200/mo per GPU—even while you game or sleep.',
        },
        {
          icon: <Sliders className="h-8 w-8" aria-hidden="true" />,
          title: 'Full Control',
          body: 'Set prices, availability windows, and usage limits.',
        },
        {
          icon: <Shield className="h-8 w-8" aria-hidden="true" />,
          title: 'Secure & Private',
          body: 'Sandboxed jobs. No access to your files.',
        },
        {
          icon: <Zap className="h-8 w-8" aria-hidden="true" />,
          title: 'Easy Setup',
          body: 'Install in ~10 minutes. No expertise required.',
        },
      ]}
      steps={[
        {
          title: 'Install rbee',
          body: 'Run one command on Windows, macOS, or Linux.',
        },
        {
          title: 'Configure Your GPUs',
          body: 'Choose pricing, availability, and usage limits in the web dashboard.',
        },
        {
          title: 'Join the Marketplace',
          body: 'Your GPUs become rentable to verified developers.',
        },
        {
          title: 'Get Paid',
          body: 'Earnings track in real time. Withdraw anytime.',
        },
      ]}
      earnings={{
        rows: [
          {
            model: 'RTX 4090',
            meta: '24GB VRAM • 450W',
            value: '€180/mo',
            note: 'at 80% utilization',
          },
          {
            model: 'RTX 4080',
            meta: '16GB VRAM • 320W',
            value: '€140/mo',
            note: 'at 80% utilization',
          },
          {
            model: 'RTX 3080',
            meta: '10GB VRAM • 320W',
            value: '€90/mo',
            note: 'at 80% utilization',
          },
        ],
        disclaimer: 'Actuals vary with demand, pricing, and availability. These are conservative estimates.',
      }}
      ctaPrimary={{
        label: 'Start Earning',
        href: '/signup',
        ariaLabel: 'Start earning with rbee',
      }}
      ctaSecondary={{
        label: 'Estimate My Payout',
        href: '#earnings-calculator',
      }}
    />
  )
}
