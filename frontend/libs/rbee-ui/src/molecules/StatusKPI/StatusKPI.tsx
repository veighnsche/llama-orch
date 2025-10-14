import { type LucideIcon } from 'lucide-react'
import { IconBox } from '@rbee/ui/molecules'
import type { IconBoxProps } from '@rbee/ui/molecules/IconBox'

interface StatusKPIProps {
  icon: LucideIcon
  color: IconBoxProps['color']
  label: string
  value: string | number
}

export function StatusKPI({ icon, color, label, value }: StatusKPIProps) {
  return (
    <div className="bg-card border border-border rounded-xl p-4 flex items-center gap-3">
      <IconBox icon={icon} color={color} size="sm" />
      <div>
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-lg font-semibold text-foreground">{value}</div>
      </div>
    </div>
  )
}
