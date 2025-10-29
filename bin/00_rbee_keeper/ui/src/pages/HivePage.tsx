// TEAM-342: Hive web interface page
// Embeds Hive's web UI (localhost:7835) in an iframe
// Disabled when Hive is not running
// Uses dynamic hiveId from URL params

import { Alert, AlertDescription, AlertTitle, Button } from "@rbee/ui/atoms";
import { PageContainer } from "@rbee/ui/molecules";
import { AlertCircle, ExternalLink, PlayCircle } from "lucide-react";
import { useParams } from "react-router-dom";
import { useSshHivesStore } from "../store/hiveStore";
import { useEffect } from "react";

// TEAM-342: Inner component that renders the iframe
function HiveIframeContent({ hiveId }: { hiveId: string }) {
  const { hives, start, fetchHiveStatus } = useSshHivesStore();

  // TEAM-342: Fetch hive status on mount
  useEffect(() => {
    fetchHiveStatus(hiveId);
  }, [hiveId, fetchHiveStatus]);

  const hive = hives.find((h) => h.host === hiveId);

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

  // TEAM-342: Hive port is 7835
  const hiveUrl = "http://localhost:7835";

  return (
    <div className="relative h-full w-full">
      <div className="absolute top-2 right-2 z-10">
        <Button variant="outline" size="sm" asChild>
          <a href={hiveUrl} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in new tab
          </a>
        </Button>
      </div>
      <iframe
        src={hiveUrl}
        className="w-full h-full border-0 rounded-lg"
        title={`${hive.host} Web Interface`}
        sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
      />
    </div>
  );
}

export default function HivePage() {
  const { hiveId } = useParams<{ hiveId: string }>();

  if (!hiveId) {
    return (
      <PageContainer
        title="Hive"
        description="Hive web interface"
        padding="default"
        className="h-full"
      >
        <div className="flex items-center justify-center h-full">
          <Alert variant="destructive" className="max-w-md">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Invalid URL</AlertTitle>
            <AlertDescription>
              <p>No hive ID specified in the URL.</p>
            </AlertDescription>
          </Alert>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer
      title={hiveId}
      description={`Web interface for hive ${hiveId}`}
      padding="default"
      className="h-full"
    >
      <HiveIframeContent hiveId={hiveId} />
    </PageContainer>
  );
}
