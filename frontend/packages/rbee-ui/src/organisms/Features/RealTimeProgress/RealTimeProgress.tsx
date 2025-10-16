import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader, SectionContainer, StatusKPI, TerminalWindow, TimelineStep } from '@rbee/ui/molecules'
import { Activity, Gauge, Timer, MemoryStick, XCircle } from 'lucide-react'

export function RealTimeProgress() {
  return (
    <SectionContainer
      title="Real‑time Progress Tracking"
      bgVariant="background"
      subtitle="Live narration of each step—model loading, token generation, resource usage—as it happens."
    >
      <div className="max-w-6xl mx-auto space-y-10">
        {/* Block 1: Live Terminal Timeline */}
        <div>
          <IconCardHeader
            icon={Activity}
            iconTone="primary"
            iconSize="md"
            title="SSE Narration Architecture"
            subtitle="Workers stream every step as Server-Sent Events—from model load to token generation."
            useCardHeader={false}
            className="mb-4"
          />

          <TerminalWindow
            title="SSE narration — worker 8001"
            ariaLabel="Server-sent events narration log"
            className="animate-in fade-in slide-in-from-bottom-2"
            footer={
              <div className="flex items-center justify-between">
                <div className="text-xs text-muted-foreground">
                  Narration → <code className="bg-muted px-1 rounded">stderr</code> · Tokens →{' '}
                  <code className="bg-muted px-1 rounded">stdout</code>
                </div>
                <div className="hidden sm:flex items-center gap-2">
                  <Badge variant="outline" className="bg-chart-3/15 text-chart-3 border-chart-3/30">OK</Badge>
                  <Badge variant="outline" className="bg-primary/15 text-primary border-primary/30">IO</Badge>
                  <Badge variant="outline" className="bg-destructive/15 text-destructive border-destructive/30">ERR</Badge>
                </div>
              </div>
            }
          >
            <div className="max-h-[340px] overflow-auto" role="log" aria-live="polite">
              <div className="text-muted-foreground animate-in fade-in duration-300">
                [00:00.00] [worker] start :8001
              </div>
              <div className="text-muted-foreground animate-in fade-in duration-300 delay-75">
                [00:00.03] [device] CUDA#1 initialized
              </div>
              <div className="text-primary animate-in fade-in duration-300 delay-150">
                [00:00.12] [loader] /models/tinyllama-q4.gguf → loading…
              </div>
              <div className="text-chart-3 animate-in fade-in duration-300 delay-200">
                [00:01.02] [loader] loaded 669MB in VRAM ✓
              </div>
              <div className="text-muted-foreground animate-in fade-in duration-300 delay-300">
                [00:01.05] [http] server ready :8001
              </div>

              <div className="mt-2 text-muted-foreground animate-in fade-in duration-300 delay-400">
                [00:01.10] [candle] inference start (18 chars)
              </div>
              <div className="text-muted-foreground animate-in fade-in duration-300 delay-500">
                [00:01.11] [tokenizer] prompt → 4 tokens
              </div>
              <div className="text-foreground animate-in fade-in duration-300 delay-600">Once upon a time…</div>
              <div className="text-chart-3 animate-in fade-in duration-300 delay-700">
                [00:01.26] [candle] generated 20 tokens (133 tok/s) ✓
              </div>
            </div>
          </TerminalWindow>
        </div>

        {/* Block 2: Stream Meter Row */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-100">
          <div className="hover:-translate-y-0.5 transition-transform">
            <StatusKPI icon={Gauge} color="chart-3" label="Throughput" value="133 tok/s" />
            <div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
              <div className="h-full w-[80%] bg-chart-3" />
            </div>
          </div>
          <div className="hover:-translate-y-0.5 transition-transform">
            <StatusKPI icon={Timer} color="primary" label="First token latency" value="150 ms" />
            <div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
              <div className="h-full w-[60%] bg-primary" />
            </div>
          </div>
          <div className="hover:-translate-y-0.5 transition-transform">
            <StatusKPI icon={MemoryStick} color="chart-2" label="VRAM used" value="669 MB" />
            <div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
              <div className="h-full w-[45%] bg-emerald-500/80" />
            </div>
          </div>
        </div>

        {/* Block 3: Cancellation Sequence Card */}
        <Card className="animate-in fade-in slide-in-from-bottom-2 delay-150">
          <CardContent className="p-6">
            <IconCardHeader
              icon={XCircle}
              iconTone="warning"
              iconSize="md"
              title="Request Cancellation"
              subtitle="Ctrl+C or API cancel stops the job, frees resources, and leaves no orphaned processes."
              useCardHeader={false}
              className="mb-4"
            />

            {/* Sequence */}
            <ol className="grid gap-3 sm:grid-cols-4 text-sm" aria-label="Cancellation sequence">
              <TimelineStep
                timestamp="t+0ms"
                title={
                  <>
                    Client sends{" "}
                    <code className="bg-muted px-1 rounded text-xs">POST /v1/cancel</code>
                  </>
                }
                description="Idempotent request."
              />
              <TimelineStep
                timestamp="t+50ms"
                title="SSE disconnect detected"
                description="Stream closes ≤ 1s."
              />
              <TimelineStep
                timestamp="t+80ms"
                title="Immediate cleanup"
                description="Stop tokens, release slot, log event."
              />
              <TimelineStep
                timestamp="t+120ms"
                title={<span className="text-chart-3">Worker idle ✓</span>}
                description="Ready for next task."
                variant="success"
              />
            </ol>
          </CardContent>
        </Card>
      </div>
    </SectionContainer>
  )
}
