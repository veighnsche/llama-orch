'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { PlaybookHeader, PlaybookItem, SectionContainer, StatusKPI, TerminalConsole } from '@rbee/ui/molecules'
import { Activity, AlertTriangle, CheckCircle2, Database, Network } from 'lucide-react'
import { useCallback } from 'react'

export function ErrorHandling() {
  const handleExpandAll = useCallback(() => {
    document.querySelectorAll<HTMLDetailsElement>('#playbook details').forEach((d) => (d.open = true))
  }, [])

  const handleCollapseAll = useCallback(() => {
    document.querySelectorAll<HTMLDetailsElement>('#playbook details').forEach((d) => (d.open = false))
  }, [])

  return (
    <SectionContainer
      title="Comprehensive Error Handling"
      bgVariant="background"
      subtitle="19+ error scenarios with clear messages and actionable fixes—no cryptic failures."
    >
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Overline badge */}
        <div className="flex justify-center">
          <Badge variant="secondary">Resiliency</Badge>
        </div>

        {/* 1. Status header (KPIs strip) */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2">
          <StatusKPI icon={CheckCircle2} color="chart-3" label="Scenarios covered" value="19+" />
          <StatusKPI icon={Network} color="primary" label="Auto-retries" value="SSH • HTTP • DL" />
          <StatusKPI icon={Database} color="chart-2" label="Fail-fast" value="Clear suggestions" />
        </div>

        {/* 2. Live timeline console */}
        <div className="animate-in fade-in slide-in-from-bottom-2 delay-100">
          <TerminalConsole
            title="error timeline — retries & jitter"
            ariaLabel="Error timeline with retry examples"
            footer={
              <div className="text-sm text-chart-3 bg-chart-3/10">
                ✓ Exponential backoff with random jitter (0.5–1.5×). SSH: 3 attempts · HTTP: 3 · Downloads: 6 with
                resume.
              </div>
            }
          >
            <div role="log" aria-live="polite">
              <div className="text-destructive">[ssh] attempt 1 → timeout (5000ms)</div>
              <div className="text-muted-foreground">retry in 0.8× backoff (1.2s jitter)</div>
              <div className="text-destructive mt-2">[ssh] attempt 2 → auth failed</div>
              <div className="text-muted-foreground">suggestion: check ~/.ssh/config or key permissions</div>
              <div className="text-primary mt-2">[http] attempt 1 → 502 Bad Gateway</div>
              <div className="text-muted-foreground">retry in 1.4× backoff (2.8s jitter)</div>
              <div className="text-chart-3 mt-2">[download] resumed from 43% — OK</div>
              <div className="text-chart-3">[worker] graceful shutdown after 30s timeout — OK</div>
            </div>
          </TerminalConsole>
        </div>

        {/* 3. Playbook accordion */}
        <div id="playbook" className="bg-card border rounded-2xl overflow-hidden">
          <PlaybookHeader
            title="Playbook"
            description="19+ scenarios across 4 categories"
            filterCategories={['Network', 'Resource', 'Model', 'Process']}
            onExpandAll={handleExpandAll}
            onCollapseAll={handleCollapseAll}
          />

          <PlaybookItem
            icon={Network}
            color="warning"
            title="Network & Connectivity"
            checkCount={4}
            severityDots={['destructive', 'primary', 'chart-3']}
            description="Detects timeouts, auth failures, and HTTP errors. Retries with exponential backoff + jitter."
            checks={[
              { severity: 'destructive', text: 'SSH connection timeout', detail: '3 retries with backoff' },
              { severity: 'destructive', text: 'SSH authentication failure', detail: '+ fix suggestions' },
              { severity: 'primary', text: 'HTTP connection failures', detail: 'auto-retry' },
              { severity: 'chart-3', text: 'Connection loss during inference', detail: 'partial results saved' },
            ]}
            footer={
              <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground/80">
                <span>
                  View logs:{' '}
                  <code className="px-1 py-0.5 rounded bg-muted text-foreground/90">~/.rbee/logs/keeper.log</code>
                </span>
                <a href="#error-timeline" className="underline hover:no-underline">
                  See timeline example
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={AlertTriangle}
            color="primary"
            title="Resource Errors"
            checkCount={4}
            severityDots={['primary', 'destructive']}
            description="Fail-fast on RAM/VRAM limits with actionable fixes and pre-download disk checks."
            checks={[
              { severity: 'primary', text: 'Insufficient RAM', detail: 'size guidance' },
              { severity: 'destructive', text: 'VRAM exhausted', detail: 'no CPU fallback' },
              { severity: 'primary', text: 'Disk space pre-check', detail: 'before download' },
              { severity: 'destructive', text: 'OOM during model load', detail: 'abort safely' },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{' '}
                <a href="/docs/errors#resource" className="underline hover:no-underline">
                  Resource errors
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={Database}
            color="chart-2"
            title="Model & Backend"
            checkCount={4}
            severityDots={['chart-2', 'primary']}
            description="Validates model presence, credentials, and backend availability before work starts."
            checks={[
              { severity: 'chart-2', text: 'Model 404', detail: 'with Hugging Face link' },
              { severity: 'primary', text: 'Private model 403', detail: 'token hint' },
              { severity: 'primary', text: 'Download failures', detail: 'resume support' },
              { severity: 'chart-2', text: 'Backend not available', detail: 'alternatives' },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{' '}
                <a href="/docs/errors#model" className="underline hover:no-underline">
                  Model & backend errors
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={Activity}
            color="chart-3"
            title="Process Lifecycle"
            checkCount={4}
            severityDots={['chart-3', 'destructive']}
            description="Observes workers from startup to shutdown; offers safe teardown and timeouts."
            checks={[
              { severity: 'destructive', text: 'Worker binary missing', detail: 'install steps' },
              { severity: 'destructive', text: 'Crash during startup', detail: 'log pointers' },
              { severity: 'chart-3', text: 'Graceful shutdown', detail: 'with active requests' },
              { severity: 'chart-3', text: 'Force-kill after 30s', detail: 'timeout' },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{' '}
                <a href="/docs/errors#lifecycle" className="underline hover:no-underline">
                  Process lifecycle errors
                </a>
              </div>
            }
          />
        </div>
      </div>
    </SectionContainer>
  )
}
