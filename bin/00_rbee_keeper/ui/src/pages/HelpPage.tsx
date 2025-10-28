// TEAM-295: Help page for Bee Keeper

import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { PageContainer } from '@rbee/ui/molecules'

export default function HelpPage() {
  return (
    <PageContainer title="Help" description="Documentation and support" padding="lg">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Getting Started</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Queen Service</h4>
                <p className="text-sm text-muted-foreground">
                  The Queen orchestrator manages your LLM infrastructure. Use the Start/Stop buttons on the dashboard to
                  control the service.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Hive Service</h4>
                <p className="text-sm text-muted-foreground">
                  The Hive service runs on localhost and manages workers. Control it using the action buttons on the
                  dashboard.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>SSH Hives</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Manage remote hives via SSH connections. Configure SSH targets in the dashboard table.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Support</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              For more information, visit the documentation or contact support.
            </p>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
