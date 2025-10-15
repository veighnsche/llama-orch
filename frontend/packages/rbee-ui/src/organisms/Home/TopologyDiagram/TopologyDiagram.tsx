import { cn } from '@rbee/ui/utils'

type Tone = 'primary' | 'accent' | 'muted'
type Tier = 'control' | 'manager' | 'execution'

export type TDNode = {
	id: string
	label: string
	icon?: React.ReactNode
	tone?: Tone
	tier: Tier
	laneIndex?: number
}

export type TDEdge = {
	id: string
	from: string
	to: string
	label?: string
	kind?: 'control' | 'telemetry'
}

export interface TopologyDiagramProps {
	nodes: TDNode[]
	edges: TDEdge[]
	showLegend?: boolean
	showLaneLabels?: boolean
	className?: string
	compact?: boolean
}

// Port component for connection points
const Port = ({ x, y }: { x: string; y: string }) => {
	return (
		<g>
			<circle cx={x} cy={y} r="1.8" className="fill-primary/10" />
			<circle cx={x} cy={y} r="0.5" className="fill-border/80" />
		</g>
	)
}

export function TopologyDiagram({
	nodes,
	edges,
	showLegend = true,
	showLaneLabels = true,
	compact,
	className,
}: TopologyDiagramProps) {
	// Y positions for each tier
	const Y = { control: '24%', manager: '50%', execution: '76%' }

	// X positions (4 fixed anchors for distribution)
	const X = ['12%', '36%', '60%', '84%']

	// Group nodes by tier
	const byTier = (tier: Tier) =>
		nodes.filter((n) => n.tier === tier).sort((a, b) => (a.laneIndex ?? 0) - (b.laneIndex ?? 0))

	// Helper to compute Bézier path
	const bez = (sx: string, sy: string, tx: string, ty: string, cy: string) =>
		`M ${sx} ${sy} C ${sx} ${cy}, ${tx} ${cy}, ${tx} ${ty}`

	// Compute node positions
	const getNodePos = (node: TDNode) => {
		const tier = byTier(node.tier)
		const idx = node.laneIndex ?? tier.indexOf(node)
		const count = tier.length

		let x: string
		if (count === 1) {
			x = '50%'
		} else if (count === 2) {
			x = idx === 0 ? '33%' : '67%'
		} else if (count === 3) {
			x = idx === 0 ? '20%' : idx === 1 ? '50%' : '80%'
		} else {
			x = X[idx] ?? '50%'
		}

		return { x, y: Y[node.tier] }
	}

	return (
		<figure
			className={cn(
				'relative isolate rounded-2xl border border-border bg-card/70 overflow-hidden',
				compact ? 'p-4 md:p-6' : 'p-6 md:p-8',
				className,
			)}
			aria-labelledby="topology-title"
			aria-describedby="topology-desc"
			role="figure"
		>
			{/* Spotlight background */}
			<div
				className="pointer-events-none absolute inset-0 bg-[radial-gradient(48rem_28rem_at_50%_10%,oklch(0.98_0_0/.06),transparent_60%)]"
				aria-hidden="true"
			/>

			<figcaption className="text-center mb-3 relative z-20">
				<span
					id="topology-title"
					className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-primary text-sm font-medium"
				>
					The Bee Architecture
				</span>
			</figcaption>

			{/* Diagram canvas with fixed aspect ratio */}
			<div className={cn('relative w-full', compact ? 'h-[400px]' : 'h-[500px]')}>
				{/* Rails */}
				<div className="absolute inset-x-0 top-[24%] h-px bg-border/50 rounded-full" aria-hidden="true" />
				<div className="absolute inset-x-0 top-[50%] h-px bg-border/40 rounded-full" aria-hidden="true" />
				<div className="absolute inset-x-0 top-[76%] h-px bg-border/40 rounded-full" aria-hidden="true" />

				{/* Rail labels */}
				{showLaneLabels && (
					<>
						<span className="absolute left-0 -translate-y-1/2 top-[24%] text-[11px] text-muted-foreground font-medium">
							Orchestrator
						</span>
						<span className="absolute left-0 -translate-y-1/2 top-[50%] text-[11px] text-muted-foreground font-medium">
							PCs / Hives
						</span>
						<span className="absolute left-0 -translate-y-1/2 top-[76%] text-[11px] text-muted-foreground font-medium">
							Devices
						</span>
					</>
				)}

				{/* SVG connectors */}
				<svg
					className="absolute inset-0 z-0 pointer-events-none"
					width="100%"
					height="100%"
					viewBox="0 0 100 100"
					preserveAspectRatio="none"
					aria-hidden="true"
					focusable="false"
				>
					<defs>
						<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" orient="auto" markerWidth="6" markerHeight="6">
							<path d="M0,0 L10,5 L0,10 z" className="fill-border/80" />
						</marker>
					</defs>
					<g fill="none" strokeLinecap="round">
						{edges.map((e) => {
							const s = nodes.find((n) => n.id === e.from)
							const t = nodes.find((n) => n.id === e.to)
							if (!s || !t) return null

							const sPos = getNodePos(s)
							const tPos = getNodePos(t)

							// Convert percentage strings to numbers (0-100 coordinate system)
							const sx = parseFloat(sPos.x)
							const sy = parseFloat(sPos.y)
							const tx = parseFloat(tPos.x)
							const ty = parseFloat(tPos.y)

							// Calculate midpoint Y for Bézier control
							const cy = (sy + ty) / 2

							const d = `M ${sx} ${sy} C ${sx} ${cy}, ${tx} ${cy}, ${tx} ${ty}`
							const edgeCls = e.kind === 'telemetry' ? 'stroke-chart-3/50 [stroke-dasharray:4_4]' : 'stroke-border'
							const strokeW = e.kind === 'telemetry' ? 0.8 : 1.0

							return (
								<g key={e.id}>
									{/* Main path */}
									<path
										d={d}
										className={edgeCls}
										strokeWidth={strokeW}
										markerEnd="url(#arrow)"
										vectorEffect="non-scaling-stroke"
									/>
									{/* Animated flow for control edges */}
									{e.kind === 'control' && (
										<path
											d={d}
											className="stroke-primary/20 motion-safe:[animation:flow_2.4s_linear_infinite]"
											strokeWidth={1.0}
											strokeDasharray="0.6 2.4"
											strokeLinecap="round"
											vectorEffect="non-scaling-stroke"
										/>
									)}
								</g>
							)
						})}

						{/* Ports at node centers */}
						{nodes.map((n) => {
							const pos = getNodePos(n)
							const x = parseFloat(pos.x)
							const y = parseFloat(pos.y)
							return <Port key={`port-${n.id}`} x={x.toString()} y={y.toString()} />
						})}
					</g>
				</svg>

				<p id="topology-desc" className="sr-only">
					Queen-rbee orchestrates work across multiple PCs. Each PC runs its own rbee-hive that manages local
					devices (GPUs, CPUs). Devices report telemetry back to their hive.
				</p>

				{/* Nodes (absolute positioned chips) */}
				<div className="relative z-10 w-full h-full">
					{nodes.map((n, idx) => {
						const pos = getNodePos(n)
						const toneClasses = {
							primary: 'bg-primary text-primary-foreground shadow-md font-semibold px-5 py-2.5',
							accent: 'bg-primary/10 text-primary border border-primary/25 font-medium px-4 py-2',
							muted: 'bg-muted text-muted-foreground border border-border font-medium px-3 py-2 text-sm',
						}

						return (
							<div
								key={n.id}
								className={cn(
									'absolute -translate-x-1/2 -translate-y-1/2 inline-flex items-center gap-2 rounded-xl whitespace-nowrap',
									toneClasses[n.tone ?? 'muted'],
									'motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-500',
								)}
								style={{
									left: pos.x,
									top: pos.y,
									animationDelay: `${idx * 80}ms`,
								}}
							>
								{n.icon && <span aria-hidden="true">{n.icon}</span>}
								<span>{n.label}</span>
							</div>
						)
					})}
				</div>

				{/* Legend */}
				{showLegend && (
					<div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex flex-wrap items-center justify-center gap-3 text-xs text-muted-foreground">
						<span className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/60 px-2.5 py-1 shadow-sm">
							<span className="w-6 h-[2px] bg-border/80 inline-block rounded-full" />
							<span>Control / Task Flow</span>
						</span>
						<span className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/60 px-2.5 py-1 shadow-sm">
							<span
								className="w-6 h-[2px] bg-chart-3/60 inline-block rounded-full"
								style={{
									backgroundImage:
										'repeating-linear-gradient(90deg,transparent,transparent 2px,currentColor 2px,currentColor 4px)',
								}}
							/>
							<span>Telemetry / Health</span>
						</span>
					</div>
				)}
			</div>
		</figure>
	)
}
