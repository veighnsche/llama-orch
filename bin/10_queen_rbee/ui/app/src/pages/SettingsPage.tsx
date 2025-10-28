// TEAM-292: Settings page
// Ported from web-ui.old

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { PageContainer } from '@rbee/ui/molecules'

export default function SettingsPage() {
  return (
    <PageContainer title="Settings" description="Configure your rbee installation">
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Queen Configuration</CardTitle>
            <CardDescription>Configure queen rbee settings</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Hive Settings</CardTitle>
            <CardDescription>Manage hive configurations</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Preferences</CardTitle>
            <CardDescription>Default model settings</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Advanced</CardTitle>
            <CardDescription>Advanced configuration options</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Coming soon...</p>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
