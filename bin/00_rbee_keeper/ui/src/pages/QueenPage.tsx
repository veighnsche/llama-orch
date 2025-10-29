// TEAM-340: Queen web interface page
// Embeds Queen's web UI (localhost:7833) in an iframe
// Disabled when Queen is not running
// Self-contained component with DaemonContainer wrapper (Rule Zero)

import { Alert, AlertDescription, AlertTitle, Button } from "@rbee/ui/atoms";
import { PageContainer } from "@rbee/ui/molecules";
import { AlertCircle, ExternalLink, PlayCircle } from "lucide-react";
import { DaemonContainer } from "../containers/DaemonContainer";
import { useQueenStore } from "../store/queenStore";

// TEAM-340: Inner component that renders after data is loaded
function QueenIframeContent() {
  const { status, start } = useQueenStore();

  if (!status?.isRunning) {
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

// TEAM-340: Self-contained component with DaemonContainer wrapper
function QueenIframe() {
  return (
    <DaemonContainer
      cacheKey="queen-iframe"
      metadata={{
        name: "Queen",
        description: "Smart API server",
      }}
      fetchFn={() => useQueenStore.getState().fetchStatus()}
    >
      <QueenIframeContent />
    </DaemonContainer>
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
