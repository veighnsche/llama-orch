// TEAM-295: Settings page for Bee Keeper

import { PageContainer } from "@rbee/ui/molecules";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";

export default function SettingsPage() {
  return (
    <PageContainer
      title="Settings"
      description="Configure rbee keeper preferences"
      padding="lg"
    >
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Application Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Settings configuration coming soon...
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Connection Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Configure Queen and Hive connection settings
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>About</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-sm">
                <span className="font-semibold">Version:</span> 0.1.0
              </p>
              <p className="text-sm">
                <span className="font-semibold">rbee keeper</span> - Local orchestration management
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  );
}
