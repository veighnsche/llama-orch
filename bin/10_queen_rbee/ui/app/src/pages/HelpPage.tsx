// TEAM-292: Help page
// Ported from web-ui.old

import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { PageContainer } from '@rbee/ui/molecules'
import { ExternalLinkIcon } from 'lucide-react'

export default function HelpPage() {
  return (
    <PageContainer title="Help & Documentation" description="Get started with rbee">
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
            <p className="text-sm text-muted-foreground mb-4">Detailed documentation for all API endpoints.</p>
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
            <p className="text-sm text-muted-foreground mb-4">Report bugs, request features, or contribute.</p>
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
            <p className="text-sm text-muted-foreground mb-4">Connect with other rbee users and developers.</p>
            <Button variant="outline" size="sm">
              Join Discord
              <ExternalLinkIcon className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
