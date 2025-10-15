import { StepsSection } from '@rbee/ui/organisms'
import { Download, Globe, Settings, Wallet } from 'lucide-react'

export function ProvidersHowItWorks() {
  return (
    <StepsSection
      id="how-it-works"
      kicker="How rbee Works"
      title="Start Earning in 4 Simple Steps"
      subtitle="No technical expertise required. Most providers finish in ~15 minutes."
      steps={[
        {
          icon: <Download className="h-8 w-8" aria-hidden="true" />,
          step: 'Step 1',
          title: 'Install rbee',
          body: 'Download and install with one command. Works on Windows, macOS, and Linux.',
          snippet: 'curl -sSL rbee.dev/install.sh | sh',
        },
        {
          icon: <Settings className="h-8 w-8" aria-hidden="true" />,
          step: 'Step 2',
          title: 'Configure Settings',
          body: 'Set your pricing, availability windows, and usage limits through the intuitive web dashboard.',
          checklist: ['Set hourly rate', 'Define availability', 'Set usage limits'],
        },
        {
          icon: <Globe className="h-8 w-8" aria-hidden="true" />,
          step: 'Step 3',
          title: 'Join Marketplace',
          body: 'Your GPUs automatically appear in the rbee marketplace. Developers can discover and rent your compute power.',
          successNote: 'Your GPUs are now live and earning.',
        },
        {
          icon: <Wallet className="h-8 w-8" aria-hidden="true" />,
          step: 'Step 4',
          title: 'Get Paid',
          body: 'Track earnings in real time. Automatic payouts to your bank or crypto wallet.',
          stats: [
            { label: 'Payout frequency', value: 'Weekly' },
            { label: 'Minimum payout', value: '€25' },
          ],
        },
      ]}
      avgTime="12 minutes"
    />
  )
}
