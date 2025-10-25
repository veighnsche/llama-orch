// TEAM-292: Help page
// Ported from web-ui.old

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms';
import { Button } from '@rbee/ui/atoms';
import { ExternalLinkIcon } from 'lucide-react';

export default function HelpPage() {
  return (
    <div className="flex-1 space-y-4">
      <div>
        <h1 className="text-3xl font-bold">Help & Documentation</h1>
        <p className="text-muted-foreground">
          Get started with rbee
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quick Start</CardTitle>
            <CardDescription>Get up and running quickly</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Learn the basics of rbee and start your first inference.
            </p>
            <Button variant="outline" size="sm">
              View Guide
              <ExternalLinkIcon className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API Reference</CardTitle>
            <CardDescription>Complete API documentation</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Detailed documentation for all API endpoints.
            </p>
            <Button variant="outline" size="sm">
              View Docs
              <ExternalLinkIcon className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>GitHub</CardTitle>
            <CardDescription>Source code and issues</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Report bugs, request features, or contribute.
            </p>
            <Button variant="outline" size="sm">
              Open GitHub
              <ExternalLinkIcon className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Community</CardTitle>
            <CardDescription>Join the discussion</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Connect with other rbee users and developers.
            </p>
            <Button variant="outline" size="sm">
              Join Discord
              <ExternalLinkIcon className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
