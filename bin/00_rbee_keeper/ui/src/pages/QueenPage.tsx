// TEAM-340: Queen web interface page
// TEAM-353: Rewritten to use query hooks (deleted DaemonContainer pattern)
// Embeds Queen's web UI (localhost:7833) in an iframe

import { Alert, AlertDescription, AlertTitle, Button } from "@rbee/ui/atoms";
import { PageContainer } from "@rbee/ui/molecules";
import { AlertCircle, ExternalLink, Loader2, PlayCircle } from "lucide-react";
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

  return (
    <div className="relative h-full w-full">
      <div className="absolute top-2 right-2 z-10">
        <Button variant="outline" size="sm" asChild>
          <a
            href="http://localhost:7833"
            target="_blank"
            rel="noopener noreferrer"
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in new tab
          </a>
        </Button>
      </div>
      <iframe
        src="http://localhost:7833"
        className="w-full h-full border-0 rounded-lg"
        title="Queen Web Interface"
        sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
      />
    </div>
  );
}

export default function QueenPage() {
  return (
    <PageContainer
      title="Queen"
      description="Web interface for the Queen orchestration service"
      padding="default"
      className="h-full"
    >
      <QueenIframe />
    </PageContainer>
  );
}
