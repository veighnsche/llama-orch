// TEAM-421: Worker details page - Shows detailed info about a specific worker
// TEAM-463: Updated comment - ModelDetailsPage split into HuggingFace/CivitAI pages
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: ArtifactDetailPageTemplate (unified with model detail pages)

import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ArtifactDetailPageTemplate, useArtifactActions } from '@rbee/ui/marketplace'
import { PageContainer } from '@rbee/ui/molecules'
import { useQuery } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { ArrowLeft, Cpu, Download, GitBranch, Package } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'

export function WorkerDetailsPage() {
  const { workerId } = useParams<{ workerId: string }>()
  const navigate = useNavigate()

  // TEAM-421: Environment-aware actions
  const actions = useArtifactActions({
    onActionSuccess: (action) => {
      console.log(`✅ ${action} started successfully`)
    },
    onActionError: (action, error) => {
      console.error(`❌ ${action} failed:`, error)
    },
  })

  // Fetch the specific worker by ID
  const {
    data: worker,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'rbee-worker', workerId],
    queryFn: async () => {
      const workers = await invoke<any[]>('marketplace_list_workers')
      const found = workers.find((w) => w.id === workerId)
      if (!found) throw new Error('Worker not found')
      return found
    },
    enabled: !!workerId,
    staleTime: 5 * 60 * 1000,
  })

  if (isLoading) {
    return (
      <PageContainer title="Loading..." description="Fetching worker details..." padding="default">
        <div className="flex items-center justify-center py-12">
          <div className="text-muted-foreground">Loading worker details...</div>
        </div>
      </PageContainer>
    )
  }

  if (error || !worker) {
    return (
      <PageContainer title="Worker Not Found" description="The requested worker could not be found" padding="default">
        <Card>
          <CardContent className="p-12 text-center">
            <Cpu className="size-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold mb-2">Worker not found</h3>
            <p className="text-muted-foreground mb-6">
              {error ? `Error: ${String(error)}` : "The worker you're looking for doesn't exist or has been removed."}
            </p>
            <Button onClick={() => navigate('/marketplace/rbee-workers')}>
              <ArrowLeft className="size-4 mr-2" />
              Back to Workers
            </Button>
          </CardContent>
        </Card>
      </PageContainer>
    )
  }

  const workerTypeConfig = {
    cpu: { label: 'CPU', variant: 'secondary' as const },
    cuda: { label: 'CUDA', variant: 'default' as const },
    metal: { label: 'Metal', variant: 'accent' as const },
    rocm: { label: 'ROCm', variant: 'accent' as const },
  }

  const typeConfig = workerTypeConfig[worker.workerType as keyof typeof workerTypeConfig]

  // Render main content cards
  const mainContent = (
    <>
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Platforms & Architecture */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Package className="size-4" />
              Platform Support
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground mb-2">Platforms</div>
              <div className="flex flex-wrap gap-2">
                {worker.platforms.map((platform: string) => (
                  <Badge key={platform} variant="outline">
                    {platform}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground mb-2">Architectures</div>
              <div className="flex flex-wrap gap-2">
                {worker.architectures.map((arch: string) => (
                  <Badge key={arch} variant="secondary">
                    {arch}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Build Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <GitBranch className="size-4" />
              Build Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Build System</div>
              <div className="text-sm">{worker.buildSystem}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">Binary Name</div>
              <div className="text-sm font-mono">{worker.binaryName}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">Install Path</div>
              <div className="text-sm font-mono text-xs">{worker.installPath}</div>
            </div>
          </CardContent>
        </Card>

        {/* Source Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Source Repository</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Type</div>
              <div className="text-sm">{worker.source.type}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">URL</div>
              <div className="text-sm font-mono text-xs break-all">{worker.source.url}</div>
            </div>
            {worker.source.branch && (
              <div>
                <div className="text-sm font-medium text-muted-foreground">Branch</div>
                <div className="text-sm">{worker.source.branch}</div>
              </div>
            )}
            {worker.source.path && (
              <div>
                <div className="text-sm font-medium text-muted-foreground">Path</div>
                <div className="text-sm font-mono text-xs">{worker.source.path}</div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Capabilities */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Capabilities</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Supported Formats</div>
              <div className="flex flex-wrap gap-2 mt-1">
                {worker.supportedFormats.map((format: string) => (
                  <Badge key={format} variant="outline">
                    {format}
                  </Badge>
                ))}
              </div>
            </div>
            {worker.maxContextLength && (
              <div>
                <div className="text-sm font-medium text-muted-foreground">Max Context Length</div>
                <div className="text-sm">{worker.maxContextLength.toLocaleString()} tokens</div>
              </div>
            )}
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Streaming:</span>{' '}
                <span className={worker.supportsStreaming ? 'text-green-600' : 'text-muted-foreground'}>
                  {worker.supportsStreaming ? '✓ Yes' : '✗ No'}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Batching:</span>{' '}
                <span className={worker.supportsBatching ? 'text-green-600' : 'text-muted-foreground'}>
                  {worker.supportsBatching ? '✓ Yes' : '✗ No'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  )

  return (
    <PageContainer title={worker.name} description={worker.description} padding="default">
      <ArtifactDetailPageTemplate
        name={worker.name}
        description={worker.description}
        backButton={{
          label: 'Back to Workers',
          onClick: () => navigate('/marketplace/rbee-workers'),
        }}
        badges={[
          { label: `v${worker.version}`, variant: 'outline' },
          { label: typeConfig.label, variant: typeConfig.variant },
          { label: worker.license, variant: 'outline' },
        ]}
        primaryAction={{
          label: actions.getButtonLabel('install'),
          icon: <Download className="size-4 mr-2" />,
          onClick: () => actions.installWorker(worker.id),
        }}
        mainContent={mainContent}
      />
    </PageContainer>
  )
}
