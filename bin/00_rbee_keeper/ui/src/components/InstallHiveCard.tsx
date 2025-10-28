// TEAM-338: Card for installing hive to a new SSH target
// Uses SshHivesDataProvider with React 19 use() hook - NO useEffect needed

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  DropdownMenuItem,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SplitButton,
} from '@rbee/ui/atoms'
import { Download, FileEdit, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { commands } from '@/generated/bindings'
import { SshHivesDataProvider } from '../containers/SshHivesContainer'
import { useCommandStore } from '../store/commandStore'
import { useSshHivesStore } from '../store/hiveStore'

// SSH Target Select Item component
function SshTargetItem({ name, subtitle }: { name: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-start">
      <span className="font-medium">{name}</span>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
    </div>
  )
}

// Inner component that reads from store
function InstallHiveContent() {
  const [selectedTarget, setSelectedTarget] = useState<string>('localhost')
  const { hives, installedHives, install, refresh } = useSshHivesStore()
  const { isExecuting } = useCommandStore()

  const handleOpenSshConfig = async () => {
    try {
      await commands.sshOpenConfig()
    } catch (error) {
      console.error('Failed to open SSH config:', error)
    }
  }

  // Filter out already installed hives
  const availableHives = hives.filter((hive) => !installedHives.includes(hive.host))
  const isLocalhostInstalled = installedHives.includes('localhost')

  return (
    <>
      <Select value={selectedTarget} onValueChange={setSelectedTarget}>
        <SelectTrigger id="install-target" className="w-full h-auto py-3">
          <SelectValue placeholder="Select target" />
        </SelectTrigger>
        <SelectContent>
          {/* Always include localhost if not installed */}
          {!isLocalhostInstalled && (
            <SelectItem value="localhost">
              <SshTargetItem name="localhost" subtitle="This machine" />
            </SelectItem>
          )}
          {/* Dynamic SSH targets (filtered) */}
          {availableHives.map((hive) => (
            <SelectItem key={hive.host} value={hive.host}>
              <SshTargetItem name={hive.host} subtitle={`${hive.user}@${hive.hostname}:${hive.port}`} />
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Install Button with Actions */}
      <SplitButton
        onClick={() => install(selectedTarget)}
        icon={<Download className="h-4 w-4" />}
        disabled={isExecuting}
        className="w-full"
        dropdownContent={
          <>
            <DropdownMenuItem onClick={refresh}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleOpenSshConfig}>
              <FileEdit className="mr-2 h-4 w-4" />
              Edit SSH Config
            </DropdownMenuItem>
          </>
        }
      >
        Install Hive
      </SplitButton>
    </>
  )
}

export function InstallHiveCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Install Hive</CardTitle>
        <CardDescription>Choose a target to install the Hive worker manager</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* SSH Target Selection */}
          <div className="space-y-2">
            <label htmlFor="install-target" className="text-sm font-medium text-foreground">
              Target
            </label>

            <SshHivesDataProvider
              fallback={
                <Select disabled>
                  <SelectTrigger id="install-target" className="w-full">
                    <SelectValue placeholder="Loading targets..." />
                  </SelectTrigger>
                </Select>
              }
            >
              <InstallHiveContent />
            </SshHivesDataProvider>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
