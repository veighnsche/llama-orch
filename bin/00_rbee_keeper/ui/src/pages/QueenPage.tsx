// TEAM-340: Queen web interface page
// TEAM-353: Rewritten to use query hooks (deleted DaemonContainer pattern)
// TEAM-352: Uses @rbee/shared-config for iframe URL (no hardcoded URLs)

import { Alert, AlertDescription, AlertTitle, Button } from "@rbee/ui/atoms";
import { AlertCircle, Loader2, PlayCircle } from "lucide-react";
import { getIframeUrl } from "@rbee/shared-config";
import { useQueen, useQueenActions } from "../store/queenQueries";

// TEAM-353: Rewritten to use query hooks
function QueenIframe() {
  const { data: queen, isLoading, error } = useQueen();
  const { start } = useQueenActions();

  // Loading state
  if (isLoading && !queen) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Error state
  if (error && !queen) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load Queen status</AlertTitle>
          <AlertDescription>
            <p>{error.message}</p>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!queen?.isRunning) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Queen is not running</AlertTitle>
          <AlertDescription className="space-y-4">
            <p>Start Queen to access the web interface.</p>
            <div className="flex gap-2">
              <Button onClick={start} size="sm">
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Queen
              </Button>
              <Button variant="outline" size="sm" asChild>
                <a href="/">Go to Services</a>
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // TEAM-352: Use shared-config for iframe URL (no hardcoded URLs)
  const isDev = import.meta.env.DEV
  const queenUrl = getIframeUrl('queen', isDev)
  
  return (
    <iframe
      src={queenUrl}
      className="w-full h-full border-0"
      title="Queen Web Interface"
      sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
      allow="cross-origin-isolated"
    />
  );
}

export default function QueenPage() {
  // No PageContainer - causes double title and padding issues
  // Iframe needs full height without extra padding
  return <QueenIframe />;
}
