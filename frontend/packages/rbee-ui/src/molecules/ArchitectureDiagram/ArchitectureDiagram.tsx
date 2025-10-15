import { type TDEdge, type TDNode, TopologyDiagram } from '@rbee/ui/organisms'

export interface ArchitectureDiagramProps {
	variant?: 'simple' | 'detailed'
	showLabels?: boolean
	className?: string
}

export function ArchitectureDiagram({ variant = 'simple', showLabels = true, className }: ArchitectureDiagramProps) {
	// Define nodes - Correct architecture:
	// - queen-rbee orchestrates everything (can be on any PC)
	// - Each PC has its own rbee-hive (resource manager)
	// - Each rbee-hive manages devices on that PC
	const nodes: TDNode[] = [
		// Control tier - Queen orchestrator
		{
			id: 'queen',
			label: 'queen‑rbee',
			icon: '👑',
			tone: 'primary',
			tier: 'control',
			laneIndex: 0,
		},
		// Manager tier - Each PC has its own rbee-hive
		{
			id: 'hive-pc1',
			label: 'Office Workstation',
			icon: '🍯',
			tone: 'accent',
			tier: 'manager',
			laneIndex: 0,
		},
		{
			id: 'hive-pc2',
			label: 'MacBook Pro M2',
			icon: '🍯',
			tone: 'accent',
			tier: 'manager',
			laneIndex: 1,
		},
		{
			id: 'hive-pc3',
			label: 'Home Server',
			icon: '🍯',
			tone: 'accent',
			tier: 'manager',
			laneIndex: 2,
		},
		// Execution tier - Devices on each PC
		// Office Workstation devices
		{
			id: 'device-rtx4090',
			label: 'RTX 4090',
			icon: '🐝',
			tone: 'muted',
			tier: 'execution',
			laneIndex: 0,
		},
		{
			id: 'device-rtx3090',
			label: 'RTX 3090',
			icon: '🐝',
			tone: 'muted',
			tier: 'execution',
			laneIndex: 0,
		},
		// MacBook Pro M2 device
		{
			id: 'device-m2max',
			label: 'M2 Max',
			icon: '🐝',
			tone: 'muted',
			tier: 'execution',
			laneIndex: 1,
		},
		// Home Server devices
		{
			id: 'device-rtx3080',
			label: 'RTX 3080',
			icon: '🐝',
			tone: 'muted',
			tier: 'execution',
			laneIndex: 2,
		},
		{
			id: 'device-cpu',
			label: 'CPU Fallback',
			icon: '🐝',
			tone: 'muted',
			tier: 'execution',
			laneIndex: 2,
		},
	]

	// Define edges - Queen orchestrates hives, hives manage their devices
	const edges: TDEdge[] = [
		// Queen to each PC's rbee-hive (control flow)
		{ id: 'queen-hive1', from: 'queen', to: 'hive-pc1', kind: 'control' },
		{ id: 'queen-hive2', from: 'queen', to: 'hive-pc2', kind: 'control' },
		{ id: 'queen-hive3', from: 'queen', to: 'hive-pc3', kind: 'control' },
		
		// Office Workstation hive to its devices
		{ id: 'hive1-rtx4090', from: 'hive-pc1', to: 'device-rtx4090', kind: 'control' },
		{ id: 'hive1-rtx3090', from: 'hive-pc1', to: 'device-rtx3090', kind: 'control' },
		
		// MacBook Pro hive to its device
		{ id: 'hive2-m2max', from: 'hive-pc2', to: 'device-m2max', kind: 'control' },
		
		// Home Server hive to its devices
		{ id: 'hive3-rtx3080', from: 'hive-pc3', to: 'device-rtx3080', kind: 'control' },
		{ id: 'hive3-cpu', from: 'hive-pc3', to: 'device-cpu', kind: 'control' },
		
		// Devices report telemetry back to their hive
		{ id: 'rtx4090-hive1', from: 'device-rtx4090', to: 'hive-pc1', kind: 'telemetry' },
		{ id: 'rtx3090-hive1', from: 'device-rtx3090', to: 'hive-pc1', kind: 'telemetry' },
		{ id: 'm2max-hive2', from: 'device-m2max', to: 'hive-pc2', kind: 'telemetry' },
		{ id: 'rtx3080-hive3', from: 'device-rtx3080', to: 'hive-pc3', kind: 'telemetry' },
		{ id: 'cpu-hive3', from: 'device-cpu', to: 'hive-pc3', kind: 'telemetry' },
	]

	return (
		<TopologyDiagram
			nodes={nodes}
			edges={edges}
			showLegend={variant === 'detailed'}
			showLaneLabels={variant === 'detailed'}
			className={className}
		/>
	)
}
