// TEAM-342: Hive web interface page
// TEAM-352: Uses @rbee/shared-config for iframe URL (no hardcoded URLs)
// Disabled when Hive is not running
// Uses dynamic hiveId from URL params

import { Alert, AlertDescription, AlertTitle, Button } from "@rbee/ui/atoms";
import { AlertCircle, ExternalLink, PlayCircle, Loader2 } from "lucide-react";
import { useParams } from "react-router-dom";
import { getIframeUrl } from "@rbee/shared-config";
import { useHive, useHiveActions } from "../store/hiveQueries";

// TEAM-367: Rewritten to use React Query
function HiveIframeContent({ hiveId }: { hiveId: string }) {
  const { data: hive, isLoading, error } = useHive(hiveId);
  const { start } = useHiveActions();

  // Loading state
  if (isLoading && !hive) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Error state
  if (error && !hive) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load hive</AlertTitle>
          <AlertDescription>
            <p>{error.message}</p>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!hive) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Hive not found</AlertTitle>
          <AlertDescription className="space-y-4">
            <p>The hive "{hiveId}" could not be found.</p>
            <Button variant="outline" size="sm" asChild>
              <a href="/">Go to Services</a>
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!hive.isInstalled) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Hive not installed</AlertTitle>
          <AlertDescription className="space-y-4">
            <p>The hive "{hiveId}" is not installed yet.</p>
            <Button variant="outline" size="sm" asChild>
              <a href="/">Go to Services</a>
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (hive.status !== "online") {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Hive is not running</AlertTitle>
          <AlertDescription className="space-y-4">
            <p>Start the hive to access the web interface.</p>
            <div className="flex gap-2">
              <Button onClick={() => start(hiveId)} size="sm">
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Hive
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
  const hiveUrl = getIframeUrl('hive', isDev);

  return (
    <iframe
      src={hiveUrl}
      className="w-full h-full border-0"
      title={`${hive.host} Web Interface`}
      sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
    />
  );
}

export default function HivePage() {
  const { hiveId } = useParams<{ hiveId: string }>();

  if (!hiveId) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Invalid URL</AlertTitle>
          <AlertDescription>
            <p>No hive ID specified in the URL.</p>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // TEAM-374: No PageContainer - causes double title and padding issues
  // Iframe needs full height without extra padding (matching QueenPage pattern)
  return <HiveIframeContent hiveId={hiveId} />;
}
