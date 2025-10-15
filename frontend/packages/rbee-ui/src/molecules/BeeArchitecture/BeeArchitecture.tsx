import { cn } from '@rbee/ui/utils'

export type WorkerNode = {
	id: string
	label: string
	kind: 'cuda' | 'metal' | 'cpu'
}

export type BeeTopology =
	| { mode: 'single-pc'; hostLabel: string; workers: WorkerNode[] }
	| { mode: 'multi-host'; hosts: Array<{ hostLabel: string; workers: WorkerNode[] }> }

export interface BeeArchitectureProps {
	topology: BeeTopology
	className?: string
}

// Atom: BeeNode - Reusable component for queen and hive boxes
interface BeeNodeProps {
	emoji: string
	label: string
	sublabel: string
	variant: 'queen' | 'hive'
	size?: 'default' | 'small'
	className?: string
}

function BeeNode({ emoji, label, sublabel, variant, size = 'default', className }: BeeNodeProps) {
	const isQueen = variant === 'queen'
	const isSmall = size === 'small'

	return (
		<div
			className={cn(
				'flex items-center gap-3 rounded-lg border',
				isQueen
					? 'border-primary/30 bg-primary/10 px-6 py-3'
					: 'border-border bg-muted px-6 py-3',
				isSmall && 'gap-2 px-4 py-2',
				className,
			)}
		>
			<span className={cn(isSmall ? 'text-xl' : 'text-2xl')} aria-hidden="true">
				{emoji}
			</span>
			<div className={isSmall ? 'text-xs' : ''}>
				<div className="font-semibold text-foreground">{label}</div>
				{sublabel && <div className="text-sm text-muted-foreground font-sans">{sublabel}</div>}
			</div>
		</div>
	)
}

// Atom: WorkerChip - Reusable component for worker nodes
interface WorkerChipProps {
	worker: WorkerNode
	index: number
}

function WorkerChip({ worker, index }: WorkerChipProps) {
	const getWorkerRing = (kind: WorkerNode['kind']) => {
		switch (kind) {
			case 'cuda':
				return 'ring-1 ring-amber-400/30'
			case 'metal':
				return 'ring-1 ring-sky-400/30'
			case 'cpu':
				return 'ring-1 ring-emerald-400/30'
		}
	}

	const getWorkerSubLabel = (kind: WorkerNode['kind']) => {
		switch (kind) {
			case 'cuda':
				return 'CUDA'
			case 'metal':
				return 'Metal'
			case 'cpu':
				return 'CPU'
		}
	}

	return (
		<div
			className={cn(
				'flex items-center gap-2 rounded-lg border border-border bg-muted px-3 py-2',
				'animate-in zoom-in-50 duration-400',
				getWorkerRing(worker.kind),
			)}
			style={{ animationDelay: `${(index + 1) * 120}ms` }}
		>
			<span className="text-xl" aria-hidden="true">
				🐝
			</span>
			<div className="text-sm">
				<div className="font-semibold text-foreground">{worker.label}</div>
				<div className="text-muted-foreground font-sans">{getWorkerSubLabel(worker.kind)}</div>
			</div>
		</div>
	)
}

export function BeeArchitecture({ topology, className }: BeeArchitectureProps) {

	const renderSinglePC = () => {
		if (topology.mode !== 'single-pc') return null

		return (
			<>
				{/* Queen */}
				<BeeNode
					emoji="👑"
					label="queen-rbee"
					sublabel="Orchestrator (brain)"
					variant="queen"
					className="animate-in fade-in slide-in-from-top-2 duration-400 delay-100"
				/>

				{/* Connector */}
				<div className="h-8 w-px bg-border" aria-hidden="true" />

				{/* Host chassis with embedded hive */}
				<div className="w-full rounded-xl border border-border bg-card/50 p-6 animate-in fade-in slide-in-from-top-2 duration-400 delay-200">
					<div className="mb-4 text-center text-sm font-medium text-muted-foreground font-sans">{topology.hostLabel}</div>
					
					{/* Hive (inside the PC) */}
					<BeeNode
						emoji="🍯"
						label="rbee-hive"
						sublabel="Resource manager"
						variant="hive"
						className="mb-4 w-fit mx-auto"
					/>

					{/* Workers */}
					<div className="flex flex-wrap justify-center gap-4">
						{topology.workers.map((worker, idx) => (
							<WorkerChip key={worker.id} worker={worker} index={idx} />
						))}
					</div>
				</div>
			</>
		)
	}

	const renderMultiHost = () => {
		if (topology.mode !== 'multi-host') return null

		return (
			<>
				{/* Queen */}
				<BeeNode
					emoji="👑"
					label="queen-rbee"
					sublabel="Orchestrator (brain)"
					variant="queen"
					className="animate-in fade-in slide-in-from-top-2 duration-400 delay-100"
				/>

				{/* Connector */}
				<div className="h-8 w-px bg-border" aria-hidden="true" />

				{/* Multiple hosts - each with its own hive */}
				<div className="grid w-full gap-6 sm:grid-cols-2 lg:grid-cols-3 animate-in fade-in slide-in-from-top-2 duration-400 delay-200">
					{topology.hosts.map((host, hostIndex) => (
						<div key={hostIndex} className="rounded-xl border border-border bg-card/50 p-4">
							<div className="mb-3 text-center text-sm font-medium text-muted-foreground font-sans">{host.hostLabel}</div>
							
							{/* Hive (inside each PC) */}
							<BeeNode
								emoji="🍯"
								label="rbee-hive"
								sublabel=""
								variant="hive"
								size="small"
								className="mb-3 w-fit mx-auto"
							/>

							{/* Workers */}
							<div className="flex flex-col gap-3">
								{host.workers.map((worker, idx) => (
									<WorkerChip key={worker.id} worker={worker} index={hostIndex * 10 + idx} />
								))}
							</div>
						</div>
					))}
				</div>
			</>
		)
	}

	const getFigcaption = () => {
		if (topology.mode === 'single-pc') {
			const workerDesc = topology.workers.map((w) => `${w.kind.toUpperCase()} ${w.label}`).join(', ')
			return `Topology diagram: queen-rbee orchestrates a PC with its own rbee-hive managing ${workerDesc}.`
		} else {
			const hostCount = topology.hosts.length
			return `Topology diagram: queen-rbee orchestrates ${hostCount} PC${hostCount > 1 ? 's' : ''}, each with its own rbee-hive managing local devices.`
		}
	}

	return (
		<figure className={cn('mx-auto mt-16 max-w-3xl rounded-xl border border-border bg-card p-8', className)}>
			<figcaption className="sr-only">{getFigcaption()}</figcaption>
			<h3 className="mb-6 text-center text-xl font-semibold text-card-foreground">The Bee Architecture</h3>
			<div className="flex flex-col items-center gap-6">
				{topology.mode === 'single-pc' ? renderSinglePC() : renderMultiHost()}
			</div>
		</figure>
	)
}
