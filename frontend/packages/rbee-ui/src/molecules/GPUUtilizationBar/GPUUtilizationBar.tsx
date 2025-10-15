export interface GPUUtilizationBarProps {
	label: string
	percentage: number
	variant?: 'primary' | 'secondary'
}

/**
 * A horizontal progress bar showing GPU or resource utilization.
 * Displays a label on the left and an animated percentage bar.
 */
export function GPUUtilizationBar({ label, percentage, variant = 'primary' }: GPUUtilizationBarProps) {
	const barColor = variant === 'primary' ? 'bg-primary' : 'bg-chart-2'
	const textColor = 'text-primary-foreground'

	return (
		<div className="flex items-center gap-3">
			<div className="flex-shrink-0 w-32 text-sm text-muted-foreground">{label}</div>
			<div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
				<div
					className={`h-full ${barColor} flex items-center justify-end pr-2 transition-[width] duration-700 ease-out motion-reduce:transition-none`}
					style={{ width: `${percentage}%` }}
				>
					<span className={`text-xs ${textColor} font-medium`}>{percentage}%</span>
				</div>
			</div>
		</div>
	)
}
