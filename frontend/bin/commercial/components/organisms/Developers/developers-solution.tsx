import { Cpu, DollarSign, Lock, Zap } from 'lucide-react'
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'

export function DevelopersSolution() {
  return (
    <SolutionSection
      id="how-it-works"
      title="Your hardware. Your models. Your control."
      subtitle="rbee orchestrates AI inference across every device in your home network. Here's how it runs on a single tower PC."
      benefits={[
        {
          icon: <DollarSign className="h-6 w-6 text-primary" aria-hidden="true" />,
          title: 'Zero ongoing costs',
          body: 'Pay only for electricity. No subscriptions or per-token fees.',
        },
        {
          icon: <Lock className="h-6 w-6 text-primary" aria-hidden="true" />,
          title: 'Complete privacy',
          body: 'Code never leaves your network. GDPR-friendly by default.',
        },
        {
          icon: <Zap className="h-6 w-6 text-primary" aria-hidden="true" />,
          title: 'You decide when to update',
          body: 'Models change only when you choose—no surprise breakages.',
        },
        {
          icon: <Cpu className="h-6 w-6 text-primary" aria-hidden="true" />,
          title: 'Use all your hardware',
          body: 'Orchestrate CUDA, Metal, and CPU. Every chip contributes.',
        },
      ]}
      topology={{
        mode: 'single-pc',
        hostLabel: 'Workstation — 2×CUDA GPU + 1×CPU',
        workers: [
          { id: 'w0', label: 'GPU 0', kind: 'cuda' },
          { id: 'w1', label: 'GPU 1', kind: 'cuda' },
          { id: 'w2', label: 'CPU 0', kind: 'cpu' },
        ],
      }}
    />
  )
}
