import { IconPlate } from '@rbee/ui/molecules'
import type { IconPlateProps } from '@rbee/ui/molecules'
import type { LucideIcon } from 'lucide-react'

interface StatusKPIProps {
	icon: LucideIcon
	color: IconPlateProps['tone']
	label: string
	value: string | number
}

export function StatusKPI({ icon, color, label, value }: StatusKPIProps) {
	return (
		<div className="bg-card border rounded-xl p-4 flex items-center gap-3">
			<IconPlate icon={icon} tone={color} size="sm" shape="rounded" />
			<div>
				<div className="text-xs text-muted-foreground">{label}</div>
				<div className="text-lg font-semibold text-foreground">{value}</div>
			</div>
		</div>
	)
}
