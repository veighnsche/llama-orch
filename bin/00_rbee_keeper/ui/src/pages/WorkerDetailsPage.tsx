// TEAM-421: Worker details page - Shows detailed info about a specific worker
// TEAM-463: Updated comment - ModelDetailsPage split into HuggingFace/CivitAI pages
// TEAM_529: Migrated to GWC adapters - replaced Tauri commands with fetchGWCWorker
// DATA LAYER: @rbee/marketplace-core GWC adapter + React Query
// PRESENTATION: ArtifactDetailPageTemplate (unified with model detail pages)

import { fetchGWCWorker } from '@rbee/marketplace-core'
import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ArtifactDetailPageTemplate, useArtifactActions } from '@rbee/ui/marketplace'
import { PageContainer } from '@rbee/ui/molecules'
import { useQuery } from '@tanstack/react-query'
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

  // TEAM_529: Fetch the specific worker by ID from GWC API
  const {
    data: rawWorker,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['marketplace', 'rbee-worker', workerId],
    queryFn: async () => {
      if (!workerId) throw new Error('Worker ID is required')
      const model = await fetchGWCWorker(workerId)
      return model
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

  if (error || !rawWorker) {
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

  // TEAM_529: Extract worker data from MarketplaceModel metadata
  const workerType = (rawWorker.type.split(' ')[0] || 'cpu') as 'cpu' | 'cuda' | 'metal' | 'rocm'
  const version = (rawWorker.metadata?.version as string) || '0.1.0'
  const implementation = (rawWorker.metadata?.implementation as string) || 'rust'
  const backends = (rawWorker.metadata?.backends as string) || workerType
  const supportedFormats = ((rawWorker.metadata?.supportedFormats as string) || '').split(', ').filter(Boolean)
  const supportsStreaming = (rawWorker.metadata?.supportsStreaming as boolean) || false
  const supportsBatching = (rawWorker.metadata?.supportsBatching as boolean) || false

  const workerTypeConfig = {
    cpu: { label: 'CPU', variant: 'secondary' as const },
    cuda: { label: 'CUDA', variant: 'default' as const },
    metal: { label: 'Metal', variant: 'accent' as const },
    rocm: { label: 'ROCm', variant: 'accent' as const },
  }

  const typeConfig = workerTypeConfig[workerType]

  // TEAM_529: Render main content cards (simplified for GWC API data)
  const mainContent = (
    <>
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Backend & Implementation */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="size-4" />
              Backend Support
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground mb-2">Supported Backends</div>
              <div className="flex flex-wrap gap-2">
                {backends.split(', ').map((backend: string) => (
                  <Badge key={backend} variant="outline">
                    {backend.toUpperCase()}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground mb-2">Implementation</div>
              <Badge variant="secondary">{implementation}</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Capabilities */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Package className="size-4" />
              Capabilities
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Supported Formats</div>
              <div className="flex flex-wrap gap-2 mt-1">
                {supportedFormats.length > 0 ? (
                  supportedFormats.map((format: string) => (
                    <Badge key={format} variant="outline">
                      {format}
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">Not specified</span>
                )}
              </div>
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Streaming:</span>{' '}
                <span className={supportsStreaming ? 'text-green-600' : 'text-muted-foreground'}>
                  {supportsStreaming ? '✓ Yes' : '✗ No'}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Batching:</span>{' '}
                <span className={supportsBatching ? 'text-green-600' : 'text-muted-foreground'}>
                  {supportsBatching ? '✓ Yes' : '✗ No'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tags */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-lg">Tags</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {rawWorker.tags.map((tag: string) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  )

  return (
    <PageContainer title={rawWorker.name} description={rawWorker.description || ''} padding="default">
      <ArtifactDetailPageTemplate
        name={rawWorker.name}
        description={rawWorker.description || ''}
        backButton={{
          label: 'Back to Workers',
          onClick: () => navigate('/marketplace/rbee-workers'),
        }}
        badges={[
          { label: `v${version}`, variant: 'outline' },
          { label: typeConfig.label, variant: typeConfig.variant },
          { label: rawWorker.license || 'Unknown', variant: 'outline' },
        ]}
        primaryAction={{
          label: actions.getButtonLabel('install'),
          icon: <Download className="size-4 mr-2" />,
          onClick: () => actions.installWorker(rawWorker.id),
        }}
        mainContent={mainContent}
      />
    </PageContainer>
  )
}
