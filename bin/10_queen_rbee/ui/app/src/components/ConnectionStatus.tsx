// Connection Status Badge
// Shows SSE heartbeat connection state

interface ConnectionStatusProps {
  connected: boolean;
}

export function ConnectionStatus({ connected }: ConnectionStatusProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-border">
      <div
        className={`h-2 w-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
      />
      <span className="text-xs">
        {connected ? "Connected" : "Disconnected"}
      </span>
    </div>
  );
}
