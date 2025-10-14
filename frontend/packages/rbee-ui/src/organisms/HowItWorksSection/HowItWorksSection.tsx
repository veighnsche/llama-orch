import { ConsoleOutput } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export type StepBlock =
	| { kind: 'terminal'; title?: string; lines: ReactNode; copyText?: string }
	| { kind: 'code'; title?: string; language?: string; lines: ReactNode; copyText?: string }
	| { kind: 'note'; content: ReactNode }

export type HowItWorksSectionProps = {
	title?: string
	subtitle?: string
	steps?: Array<{
		label: string
		number?: number
		block?: StepBlock
	}>
	id?: string
	className?: string
}

const DEFAULT_STEPS: Array<{
	label: string
	number?: number
	block?: StepBlock
}> = [
	{
		label: 'Install rbee',
		block: {
			kind: 'terminal',
			title: 'terminal',
			lines: (
				<>
					<div>curl -sSL https://rbee.dev/install.sh | sh</div>
					<div className="text-slate-400">rbee-keeper daemon start</div>
				</>
			),
			copyText: 'curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start',
		},
	},
	{
		label: 'Add your machines',
		block: {
			kind: 'terminal',
			title: 'terminal',
			lines: (
				<>
					<div>rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10</div>
					<div className="text-slate-400">rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20</div>
				</>
			),
			copyText:
				'rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20',
		},
	},
	{
		label: 'Configure your IDE',
		block: {
			kind: 'terminal',
			title: 'terminal',
			lines: (
				<>
					<div>
						<span className="text-blue-400">export</span> OPENAI_API_BASE=http://localhost:8080/v1
					</div>
					<div className="text-slate-400"># OpenAI-compatible endpoint — works with Zed & Cursor</div>
				</>
			),
			copyText: 'export OPENAI_API_BASE=http://localhost:8080/v1',
		},
	},
	{
		label: 'Build AI agents',
		block: {
			kind: 'code',
			title: 'TypeScript',
			language: 'ts',
			lines: (
				<>
					<div>
						<span className="text-purple-400">import</span> {'{'} invoke {'}'}{' '}
						<span className="text-purple-400">from</span>{' '}
						<span className="text-amber-400">&apos;@llama-orch/utils&apos;</span>;
					</div>
					<div className="mt-2">
						<span className="text-blue-400">const</span> code = <span className="text-blue-400">await</span>{' '}
						<span className="text-green-400">invoke</span>
						{'({'}
					</div>
					<div className="pl-4">
						prompt: <span className="text-amber-400">&apos;Generate API from schema&apos;</span>,
					</div>
					<div className="pl-4">
						model: <span className="text-amber-400">&apos;llama-3.1-70b&apos;</span>
					</div>
					<div>{'});'}</div>
				</>
			),
			copyText:
				"import { invoke } from '@llama-orch/utils';\n\nconst code = await invoke({\n  prompt: 'Generate API from schema',\n  model: 'llama-3.1-70b'\n});",
		},
	},
]

export function HowItWorksSection({
	title = 'From zero to AI infrastructure in 15 minutes',
	subtitle,
	steps = DEFAULT_STEPS,
	id,
	className,
}: HowItWorksSectionProps) {
	return (
		<section
			id={id}
			className={cn('border-b border-border bg-secondary py-24', 'animate-in fade-in-50 duration-500', className)}
		>
			<div className="mx-auto max-w-7xl px-6 lg:px-8">
				{/* Header */}
				<div className="mx-auto max-w-2xl text-center">
					<h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">{title}</h2>
					{subtitle && <p className="text-balance text-lg leading-relaxed text-muted-foreground">{subtitle}</p>}
				</div>

				{/* Steps */}
				<div className="mx-auto mt-16 max-w-4xl space-y-12">
					{steps.map((step, index) => {
						const stepNumber = step.number ?? index + 1
						return (
							<div
								key={index}
								className={cn(
									'flex gap-6 animate-in slide-in-from-bottom-2 fade-in duration-500',
									'border-t border-border/60 pt-8 sm:border-0 sm:pt-0',
								)}
								style={{ animationDelay: `${index * 120}ms` }}
							>
								{/* Step badge */}
								<div
									className="grid h-12 w-12 flex-shrink-0 place-content-center rounded-lg bg-primary text-xl font-bold text-primary-foreground"
									aria-hidden="true"
								>
									{stepNumber}
								</div>

								{/* Step content */}
								<div className="flex-1">
									<h3 className="mb-3 text-xl font-semibold text-card-foreground">{step.label}</h3>

									{/* Render block */}
									{step.block && (
										<>
											{step.block.kind === 'terminal' && (
												<ConsoleOutput
													showChrome
													title={step.block.title || 'terminal'}
													background="dark"
													copyable
													copyText={step.block.copyText}
												>
													{step.block.lines}
												</ConsoleOutput>
											)}

											{step.block.kind === 'code' && (
												<ConsoleOutput
													showChrome
													title={step.block.title || step.block.language || 'code'}
													background="dark"
													copyable
													copyText={step.block.copyText}
												>
													{step.block.lines}
												</ConsoleOutput>
											)}

											{step.block.kind === 'note' && (
												<div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
													{step.block.content}
												</div>
											)}
										</>
									)}
								</div>
							</div>
						)
					})}
				</div>
			</div>
		</section>
	)
}
