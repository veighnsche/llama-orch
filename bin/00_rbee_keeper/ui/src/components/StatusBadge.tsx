// TEAM-294: Status badge component
// Displays connection status with colored badge and dot indicator

interface StatusBadgeProps {
  status: "online" | "offline" | "unknown";
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const styles = {
    online:
      "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-green-500/10 text-green-500",
    offline:
      "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-red-500/10 text-red-500",
    unknown:
      "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium bg-gray-500/10 text-gray-500",
  };

  const dotStyles = {
    online: "mr-1.5 h-1.5 w-1.5 rounded-full bg-green-500",
    offline: "mr-1.5 h-1.5 w-1.5 rounded-full bg-red-500",
    unknown: "mr-1.5 h-1.5 w-1.5 rounded-full bg-gray-500",
  };

  return (
    <span className={styles[status]}>
      <span className={dotStyles[status]} />
      {status}
    </span>
  );
}
