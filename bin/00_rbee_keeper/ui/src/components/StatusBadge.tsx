// TEAM-294: Status badge component
// Displays connection status with colored badge and dot indicator
// TEAM-338: Made clickable button to manually refresh status

interface StatusBadgeProps {
  status: 'running' | 'stopped' | 'unknown'
  onClick?: () => void
  isLoading?: boolean
}

export function StatusBadge({ status, onClick, isLoading = false }: StatusBadgeProps) {
  const styles = {
    running:
      'inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-green-500/10 text-green-500 transition-colors',
    stopped:
      'inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-red-500/10 text-red-500 transition-colors',
    unknown:
      'inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-gray-500/10 text-gray-500 transition-colors',
  }

  const dotStyles = {
    running: 'mr-1.5 h-1.5 w-1.5 rounded-full bg-green-500',
    stopped: 'mr-1.5 h-1.5 w-1.5 rounded-full bg-red-500',
    unknown: 'mr-1.5 h-1.5 w-1.5 rounded-full bg-gray-500',
  }

  const labels = {
    running: 'Running',
    stopped: 'Stopped',
    unknown: 'Unknown',
  }

  const Component = onClick ? 'button' : 'span'
  const buttonProps = onClick
    ? {
        onClick,
        disabled: isLoading,
        className: `${styles[status]} ${
          !isLoading ? 'hover:bg-opacity-20 cursor-pointer active:scale-95' : 'opacity-50 cursor-not-allowed'
        }`,
        type: 'button' as const,
      }
    : {
        className: styles[status],
      }

  return (
    <Component {...buttonProps}>
      <span className={`${dotStyles[status]} ${isLoading ? 'animate-pulse' : ''}`} />
      {labels[status]}
    </Component>
  )
}
