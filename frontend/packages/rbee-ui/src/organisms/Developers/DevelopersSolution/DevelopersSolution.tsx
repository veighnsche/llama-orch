import { SolutionSection } from '@rbee/ui/organisms'
import { Cpu, DollarSign, Lock, Zap } from 'lucide-react'

export function DevelopersSolution() {
	return (
		<SolutionSection
			id="how-it-works"
			kicker="How rbee Works"
			title="Your Hardware. Your Models. Your Control."
			subtitle="rbee orchestrates AI inference across every device in your home network, turning idle hardware into a private, OpenAI-compatible AI platform."
			features={[
				{
					icon: <DollarSign className="h-8 w-8" aria-hidden="true" />,
					title: 'Zero Ongoing Costs',
					body: 'Pay only for electricity. No subscriptions or per-token fees.',
				},
				{
					icon: <Lock className="h-8 w-8" aria-hidden="true" />,
					title: 'Complete Privacy',
					body: 'Code never leaves your network. GDPR-friendly by default.',
				},
				{
					icon: <Zap className="h-8 w-8" aria-hidden="true" />,
					title: 'You Decide When to Update',
					body: 'Models change only when you choose—no surprise breakages.',
				},
				{
					icon: <Cpu className="h-8 w-8" aria-hidden="true" />,
					title: 'Use All Your Hardware',
					body: 'Orchestrate CUDA, Metal, and CPU. Every chip contributes.',
				},
			]}
			steps={[
				{
					title: 'Install rbee',
					body: 'Run one command on Windows, macOS, or Linux.',
				},
				{
					title: 'Add Your Hardware',
					body: 'rbee auto-detects GPUs and CPUs across your network.',
				},
				{
					title: 'Download Models',
					body: 'Pull models from Hugging Face or load local GGUF files.',
				},
				{
					title: 'Start Building',
					body: 'OpenAI-compatible API. Drop-in replacement for your existing code.',
				},
			]}
			ctaPrimary={{
				label: 'Get Started',
				href: '/getting-started',
			}}
			ctaSecondary={{
				label: 'View Documentation',
				href: '/docs',
			}}
		/>
	)
}
