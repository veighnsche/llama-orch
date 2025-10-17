import { Badge, CacheLayer, DiagnosticGrid, DistributedNodes, ProgressTimeline } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import type {
  AdditionalFeaturesGridProps,
  CrossNodeOrchestrationProps,
  EmailCaptureProps,
  ErrorHandlingTemplateProps,
  FeaturesTabsProps,
  IntelligentModelManagementProps,
  MultiBackendGpuTemplateProps,
  RealTimeProgressProps,
  SecurityIsolationProps,
} from '@rbee/ui/templates'
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  Code,
  Cpu,
  Database,
  Gauge,
  MemoryStick,
  Network,
  Shield,
  Terminal,
  Timer,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Features Hero ===

/**
 * Features Hero - Above-the-fold hero section
 * Note: FeaturesHero takes no props - it's a self-contained organism
 */

// === Features Tabs ===

/**
 * Features tabs content - Four core capabilities (API, GPU, Scheduler, SSE)
 * with interactive examples
 */
export const featuresFeaturesTabsProps: FeaturesTabsProps = {
  title: 'Core capabilities',
  description: 'Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.',
  tabs: [
    {
      value: 'api',
      icon: <Code className="size-6" />,
      label: 'OpenAI-Compatible',
      mobileLabel: 'API',
      subtitle: 'Drop-in API',
      badge: 'Drop-in',
      description: 'Swap endpoints, keep your code. Works with Zed, Cursor, Continue—any OpenAI client.',
      content: (
        <CodeBlock
          code={`# Before: OpenAI
export OPENAI_API_KEY=sk-...

# After: rbee (same code)
export OPENAI_API_BASE=http://localhost:8080/v1`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Drop-in replacement. Point to localhost.',
        variant: 'success',
      },
      benefits: [{ text: 'No vendor lock-in' }, { text: 'Use your models + GPUs' }, { text: 'Keep existing tooling' }],
    },
    {
      value: 'gpu',
      icon: <Cpu className="size-6" />,
      label: 'Multi-GPU',
      mobileLabel: 'GPU',
      subtitle: 'Use every GPU',
      badge: 'Scale',
      description: 'Run across CUDA, Metal, and CPU backends. Use every GPU across your network.',
      content: (
        <div className="space-y-3">
          <GPUUtilizationBar label="RTX 4090 #1" percentage={92} />
          <GPUUtilizationBar label="RTX 4090 #2" percentage={88} />
          <GPUUtilizationBar label="M2 Ultra" percentage={76} />
          <GPUUtilizationBar label="CPU Backend" percentage={34} variant="secondary" />
        </div>
      ),
      highlight: {
        text: 'Higher throughput by saturating all devices.',
        variant: 'success',
      },
      benefits: [
        { text: 'Bigger models fit' },
        { text: 'Lower latency under load' },
        { text: 'No single-machine bottleneck' },
      ],
    },
    {
      value: 'scheduler',
      icon: <Gauge className="size-6" />,
      label: 'Programmable scheduler (Rhai)',
      mobileLabel: 'Rhai',
      subtitle: 'Route with Rhai',
      badge: 'Control',
      description: 'Write routing rules. Send 70B to multi-GPU, images to CUDA, everything else to cheapest.',
      content: (
        <CodeBlock
          code={`// Custom routing logic
if task.model.contains("70b") {
  route_to("multi-gpu-cluster")
}
else if task.type == "image" {
  route_to("cuda-only")
}
else {
  route_to("cheapest")
}`}
          language="rust"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Optimize for cost, latency, or compliance—your rules.',
        variant: 'primary',
      },
      benefits: [{ text: 'Deterministic routing' }, { text: 'Policy & compliance ready' }, { text: 'Easy to evolve' }],
    },
    {
      value: 'sse',
      icon: <Zap className="size-6" />,
      label: 'Task-based API with SSE',
      mobileLabel: 'SSE',
      subtitle: 'Live job stream',
      badge: 'Observe',
      description: 'See model loading, token generation, and costs stream in as they happen.',
      content: (
        <TerminalWindow
          showChrome={false}
          copyable={true}
          copyText={`→ event: task.created
{ "id": "task_123", "status": "pending" }

→ event: model.loading
{ "progress": 0.45, "eta": "2.1s" }

→ event: token.generated
{ "token": "const", "total": 1 }

→ event: token.generated
{ "token": " api", "total": 2 }`}
        >
          <div className="space-y-2" role="log" aria-live="polite">
            <div role="status">
              <div className="text-muted-foreground">→ event: task.created</div>
              <div className="pl-4">{'{ "id": "task_123", "status": "pending" }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: model.loading</div>
              <div className="pl-4">{'{ "progress": 0.45, "eta": "2.1s" }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: token.generated</div>
              <div className="pl-4">{'{ "token": "const", "total": 1 }'}</div>
            </div>
            <div role="status">
              <div className="text-muted-foreground mt-2">→ event: token.generated</div>
              <div className="pl-4">{'{ "token": " api", "total": 2 }'}</div>
            </div>
          </div>
        </TerminalWindow>
      ),
      highlight: {
        text: 'Full visibility for every inference job.',
        variant: 'default',
      },
      benefits: [{ text: 'Faster debugging' }, { text: 'UX you can trust' }, { text: 'Accurate cost tracking' }],
    },
  ],
  defaultTab: 'api',
}

// === Cross-Node Orchestration ===

/**
 * Cross-Node Orchestration container - Layout configuration
 */
export const crossNodeOrchestrationContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: <Badge variant="secondary">Distributed execution</Badge>,
  title: 'Cross-Pool Orchestration',
  description:
    'Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in your pool.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <DistributedNodes className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Cross-Node Orchestration - Distributed execution across network
 */
export const crossNodeOrchestrationProps: CrossNodeOrchestrationProps = {
  terminalContent: (
    <>
      <div className="text-muted-foreground"># Add a remote machine to your pool</div>
      <div className="text-chart-3 mt-2">
        $ rbee-keeper setup add-node \
        <br />
        {'  '}--name workstation \
        <br />
        {'  '}--ssh-host workstation.home.arpa \
        <br />
        {'  '}--ssh-user vince \
        <br />
        {'  '}--ssh-key ~/.ssh/id_ed25519
      </div>
      <div className="text-muted-foreground mt-4"># Run inference on that machine</div>
      <div className="text-chart-3 mt-2">
        $ rbee-keeper infer --node workstation \
        <br />
        {'  '}--model hf:meta-llama/Llama-3.1-8B \
        <br />
        {'  '}--prompt &quot;write a short story&quot;
      </div>
    </>
  ),
  terminalCopyText: `# Add a remote machine to your pool\n$ rbee-keeper setup add-node \\\n  --name workstation \\\n  --ssh-host workstation.home.arpa \\\n  --ssh-user vince \\\n  --ssh-key ~/.ssh/id_ed25519\n\n# Run inference on that machine\n$ rbee-keeper infer --node workstation \\\n  --model hf:meta-llama/Llama-3.1-8B \\\n  --prompt "write a short story"`,
  benefits: [
    {
      icon: <CheckCircle2 className="size-6" />,
      title: 'SSH Tunneling',
      description: 'Secure connections over SSH.',
    },
    {
      icon: <CheckCircle2 className="size-6" />,
      title: 'Auto Shutdown',
      description: 'Workers exit cleanly after tasks.',
    },
    {
      icon: <CheckCircle2 className="size-6" />,
      title: 'Minimal Footprint',
      description: 'No persistent daemons on nodes.',
    },
  ],
  diagramNodes: [
    { name: 'queen-rbee', label: 'Orchestrator', tone: 'primary' },
    { name: 'rbee-hive', label: 'Pool manager', tone: 'chart-2' },
    { name: 'worker-rbee', label: 'Inference worker', tone: 'chart-3' },
  ],
  diagramArrows: [
    { label: 'SSH', indent: 'pl-8' },
    { label: 'Spawns', indent: 'pl-16' },
  ],
  legendItems: [{ label: 'On-demand start' }, { label: 'Clean shutdown' }, { label: 'No daemon drift' }],
  provisioningTitle: 'Automatic Worker Provisioning',
  provisioningSubtitle: 'rbee spawns workers via SSH on demand and shuts them down cleanly. No manual daemons.',
}

// TO BE CONTINUED - File is getting large, will split into multiple write operations

// === Intelligent Model Management ===

/**
 * Intelligent Model Management container - Layout configuration
 */
export const intelligentModelManagementContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: <Badge variant="secondary">Provision • Cache • Validate</Badge>,
  title: 'Intelligent Model Management',
  description: 'Automatic model provisioning, caching, and validation. Download once; use everywhere.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <CacheLayer className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '5xl',
  align: 'center',
}

/**
 * Intelligent Model Management - Auto-provisioning and caching
 */
export const intelligentModelManagementProps: IntelligentModelManagementProps = {
  catalogTitle: 'Automatic Model Catalog',
  catalogDescription:
    'Request any model from Hugging Face. rbee downloads, verifies checksums, and caches locally so you never fetch the same model twice.',
  timelineContent: (
    <div className="space-y-3">
      <div className="text-muted-foreground animate-in fade-in duration-300">
        → [model-provisioner] Downloading from Hugging Face…
      </div>
      <div className="animate-in fade-in duration-300 delay-75">
        <div className="text-foreground">→ [model-provisioner] 20% (1 MB / 5 MB)</div>
        <div className="h-2 w-full bg-muted rounded-full overflow-hidden mt-2">
          <div className="h-full bg-chart-3 animate-in grow-in origin-left" style={{ width: '20%' }} />
        </div>
      </div>
      <div className="animate-in fade-in duration-300 delay-150">
        <div className="text-foreground">→ [model-provisioner] 100% (5 MB / 5 MB)</div>
        <div className="h-2 w-full bg-muted rounded-full overflow-hidden mt-2">
          <div className="h-full bg-chart-3 animate-in grow-in origin-left" style={{ width: '100%' }} />
        </div>
      </div>
      <div className="text-chart-3 animate-in fade-in duration-300 delay-200">
        → [model-provisioner] ✅ Saved to /models/tinyllama-q4.gguf
      </div>
      <div className="text-muted-foreground animate-in fade-in duration-300 delay-300">
        → [model-provisioner] Verifying SHA256…
      </div>
      <div className="text-chart-3 animate-in fade-in duration-300 delay-400">
        → [model-provisioner] ✅ Checksum verified
      </div>
    </div>
  ),
  modelSources: [
    { title: 'Hugging Face', example: 'hf:meta-llama/Llama-3.1-8B' },
    { title: 'Local GGUF', example: 'file:./models/llama-3.1.gguf' },
    { title: 'HTTP URL', example: 'https://example.com/model.gguf' },
  ],
  preflightTitle: 'Resource Preflight Checks',
  preflightDescription:
    'Before any load, rbee validates RAM, VRAM, and disk capacity to fail fast with clear errors—no mystery crashes.',
  resourceChecks: [
    {
      title: 'RAM check',
      description: 'Requires available RAM ≥ model size × 1.2',
    },
    {
      title: 'VRAM check',
      description: 'Sufficient GPU VRAM for selected backend',
    },
    {
      title: 'Disk space',
      description: 'Free space verified before download',
    },
    {
      title: 'Backend availability',
      description: 'CUDA • Metal • CPU presence',
    },
  ],
  alertMessage: 'Prevents failed loads by validating resources up front.',
}

// === Multi-Backend GPU ===

/**
 * Multi-Backend GPU container - Layout configuration
 */
export const multiBackendGpuContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Multi-Backend GPU Support',
  description:
    'CUDA, Metal, and CPU backends with explicit device selection. No silent fallbacks—you control the hardware.',
  background: {

    variant: 'background',

  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Multi-Backend GPU - CUDA, Metal, CPU support
 */
export const multiBackendGpuProps: MultiBackendGpuTemplateProps = {
  policyTitle: 'GPU FAIL FAST policy',
  policySubtitle: 'No silent fallbacks. Clear errors with suggestions. You choose the backend.',
  prohibitedBadges: [
    { label: 'No GPU→CPU fallback', variant: 'destructive' },
    { label: 'No graceful degradation', variant: 'destructive' },
    { label: 'No implicit CPU reroute', variant: 'destructive' },
  ],
  whatHappensBadges: [
    { label: 'Fail fast (exit 1)', variant: 'success' },
    { label: 'Helpful error message', variant: 'success' },
    { label: 'Explicit backend selection', variant: 'success' },
  ],
  errorTitle: 'Insufficient VRAM: need 4000 MB, have 2000 MB',
  errorSuggestions: [
    'Use smaller quantized model (Q4_K_M instead of Q8_0)',
    'Try CPU backend explicitly (--backend cpu)',
    'Free VRAM by closing other applications',
  ],
  terminalTitle: 'rbee-hive detect — workstation.home.arpa',
  terminalContent: <div className="text-chart-3">rbee-hive detect</div>,
  backendDetections: [
    { label: 'cuda × 2', variant: 'primary' },
    { label: 'cpu × 1', variant: 'muted' },
    { label: 'metal × 0', variant: 'success' },
  ],
  totalDevices: 3,
  terminalFooter: 'Cached in the registry for fast lookups and policy routing.',
  featureCards: [
    {
      icon: <Cpu className="size-6" />,
      title: 'Detection',
      description: 'Scans CUDA, Metal, CPU and counts devices.',
    },
    {
      icon: <CheckCircle2 className="size-6" />,
      title: 'Explicit selection',
      description: 'Choose backend & device—no surprises.',
    },
    {
      icon: <AlertTriangle className="size-6" />,
      title: 'Helpful suggestions',
      description: 'Actionable fixes on error.',
    },
  ],
}

// === Error Handling ===

/**
 * Error Handling container - Layout configuration
 */
export const errorHandlingContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: <Badge variant="secondary">Resiliency</Badge>,
  title: 'Comprehensive Error Handling',
  description: '19+ error scenarios with clear messages and actionable fixes—no cryptic failures.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <DiagnosticGrid className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Error Handling - Comprehensive error scenarios
 */
export const errorHandlingProps: ErrorHandlingTemplateProps = {
  statusKPIs: [
    {
      icon: <CheckCircle2 className="size-6" />,
      color: 'chart-3',
      label: 'Scenarios covered',
      value: '19+',
    },
    {
      icon: <Network className="size-6" />,
      color: 'primary',
      label: 'Auto-retries',
      value: 'SSH • HTTP • DL',
    },
    {
      icon: <Database className="size-6" />,
      color: 'chart-2',
      label: 'Fail-fast',
      value: 'Clear suggestions',
    },
  ],
  terminalContent: (
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
  ),
  terminalFooter: (
    <div className="flex items-center gap-2 text-sm text-chart-3 font-sans">
      <Check className="h-4 w-4" />
      <span>
        Exponential backoff with random jitter (0.5–1.5×). SSH: 3 attempts · HTTP: 3 · Downloads: 6 with resume.
      </span>
    </div>
  ),
  playbookCategories: [
    {
      icon: <Network className="size-6" />,
      color: 'warning',
      title: 'Network & Connectivity',
      checkCount: 4,
      severityDots: ['destructive', 'primary', 'chart-3'],
      description: 'Detects timeouts, auth failures, and HTTP errors. Retries with exponential backoff + jitter.',
      checks: [
        {
          severity: 'destructive',
          title: 'SSH connection timeout',
          meaning: "Remote host didn't respond within threshold; likely egress or firewall.",
          actionLabel: 'Retry with jitter',
          href: '#retry',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'destructive',
          title: 'SSH authentication failure',
          meaning: 'Keys or agents rejected by server.',
          actionLabel: 'Open fix steps',
          href: '#fix',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'primary',
          title: 'HTTP connection failures',
          meaning: 'Auto-retry on transient TCP resets.',
          actionLabel: 'Enable auto-retry',
          href: '#enable',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'chart-3',
          title: 'Connection loss during inference',
          meaning: 'Stream dropped; partial results were saved.',
          actionLabel: 'Resume stream',
          href: '#resume',
          guideLabel: 'Timeline',
          guideHref: '#timeline',
        },
      ],
      footer: (
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground/80">
          <span>
            View logs: <code className="px-1 py-0.5 rounded bg-muted text-foreground/90">~/.rbee/logs/keeper.log</code>
          </span>
          <a href="#error-timeline" className="underline hover:no-underline">
            See timeline example
          </a>
        </div>
      ),
    },
    {
      icon: <AlertTriangle className="size-6" />,
      color: 'primary',
      title: 'Resource Errors',
      checkCount: 4,
      severityDots: ['primary', 'destructive'],
      description: 'Fail-fast on RAM/VRAM limits with actionable fixes and pre-download disk checks.',
      checks: [
        {
          severity: 'primary',
          title: 'Insufficient RAM',
          meaning: 'Process exceeds available memory; swap thrashing.',
          actionLabel: 'Lower batch / increase memory',
          href: '#memory',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'destructive',
          title: 'VRAM exhausted',
          meaning: 'GPU out of memory; no CPU fallback configured.',
          actionLabel: 'Use smaller precision',
          href: '#precision',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'primary',
          title: 'Disk space pre-check',
          meaning: 'Download blocked to prevent low-disk failures.',
          actionLabel: 'Choose cache path',
          href: '#cache',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'destructive',
          title: 'OOM during model load',
          meaning: 'Abort safely before corrupted state.',
          actionLabel: 'Stream weights',
          href: '#stream',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
      ],
      footer: (
        <div className="mt-4 text-xs text-muted-foreground/80">
          Docs:{' '}
          <a href="/docs/errors#resource" className="underline hover:no-underline">
            Resource errors
          </a>
        </div>
      ),
    },
    {
      icon: <Database className="size-6" />,
      color: 'chart-2',
      title: 'Model & Backend',
      checkCount: 4,
      severityDots: ['chart-2', 'primary'],
      description: 'Validates model presence, credentials, and backend availability before work starts.',
      checks: [
        {
          severity: 'chart-2',
          title: 'Model 404 (Hugging Face link)',
          meaning: 'Requested model not found or renamed.',
          actionLabel: 'Select available model',
          href: '#select',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'primary',
          title: 'Private model 403',
          meaning: 'Token lacks permission for repository.',
          actionLabel: 'Fix token scope',
          href: '#token',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'primary',
          title: 'Download failures (resume support)',
          meaning: 'Interrupted download with resumable chunks.',
          actionLabel: 'Resume now',
          href: '#resume',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'chart-2',
          title: 'Backend not available',
          meaning: 'Health probe failing; show alternatives.',
          actionLabel: 'Switch endpoint',
          href: '#switch',
          guideLabel: 'Alternatives',
          guideHref: '#alternatives',
        },
      ],
      footer: (
        <div className="mt-4 text-xs text-muted-foreground/80">
          Docs:{' '}
          <a href="/docs/errors#model" className="underline hover:no-underline">
            Model & backend errors
          </a>
        </div>
      ),
    },
    {
      icon: <Activity className="size-6" />,
      color: 'chart-3',
      title: 'Process Lifecycle',
      checkCount: 4,
      severityDots: ['chart-3', 'destructive'],
      description: 'Observes workers from startup to shutdown; offers safe teardown and timeouts.',
      checks: [
        {
          severity: 'destructive',
          title: 'Worker binary missing',
          meaning: 'Install steps incomplete; worker cannot spawn.',
          actionLabel: 'Run installer',
          href: '#install',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'destructive',
          title: 'Crash during startup',
          meaning: 'Read early log pointers for root cause.',
          actionLabel: 'Open logs',
          href: '#logs',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'chart-3',
          title: 'Graceful shutdown',
          meaning: 'Drain active requests before exit.',
          actionLabel: 'Send SIGTERM',
          href: '#sigterm',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
        {
          severity: 'chart-3',
          title: 'Force-kill after 30s',
          meaning: 'Timeout guard to prevent hung exits.',
          actionLabel: 'Adjust timeout',
          href: '#timeout',
          guideLabel: 'Guide',
          guideHref: '#guide',
        },
      ],
      footer: (
        <div className="mt-4 text-xs text-muted-foreground/80">
          Docs:{' '}
          <a href="/docs/errors#lifecycle" className="underline hover:no-underline">
            Process lifecycle errors
          </a>
        </div>
      ),
    },
  ],
}

// === Real-Time Progress ===

/**
 * Real-Time Progress container - Layout configuration
 */
export const realTimeProgressContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Real‑time Progress Tracking',
  description: 'Live narration of each step—model loading, token generation, resource usage—as it happens.',
  background: {
    variant: 'background',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <ProgressTimeline className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Real-Time Progress - SSE narration and metrics
 */
export const realTimeProgressProps: RealTimeProgressProps = {
  narrationTitle: 'SSE Narration Architecture',
  narrationSubtitle: 'Workers stream every step as Server-Sent Events—from model load to token generation.',
  terminalTitle: 'SSE narration — worker 8001',
  terminalAriaLabel: 'Server-sent events narration log',
  terminalContent: (
    <div className="max-h-[340px] overflow-auto" role="log" aria-live="polite">
      <div className="text-muted-foreground animate-in fade-in duration-300">[00:00.00] [worker] start :8001</div>
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
  ),
  terminalFooter: (
    <div className="flex items-center justify-between">
      <div className="text-xs text-muted-foreground">
        Narration → <code className="bg-muted px-1 rounded">stderr</code> · Tokens →{' '}
        <code className="bg-muted px-1 rounded">stdout</code>
      </div>
      <div className="hidden sm:flex items-center gap-2">
        <Badge variant="outline" className="bg-chart-3/15 text-chart-3 border-chart-3/30">
          OK
        </Badge>
        <Badge variant="outline" className="bg-primary/15 text-primary border-primary/30">
          IO
        </Badge>
        <Badge variant="outline" className="bg-destructive/15 text-destructive border-destructive/30">
          ERR
        </Badge>
      </div>
    </div>
  ),
  metricKPIs: [
    {
      icon: <Gauge className="size-6" />,
      color: 'chart-3',
      label: 'Throughput',
      value: '133 tok/s',
      progressPercentage: 80,
    },
    {
      icon: <Timer className="size-6" />,
      color: 'primary',
      label: 'First token latency',
      value: '150 ms',
      progressPercentage: 60,
    },
    {
      icon: <MemoryStick className="size-6" />,
      color: 'chart-2',
      label: 'VRAM used',
      value: '669 MB',
      progressPercentage: 45,
    },
  ],
  cancellationTitle: 'Request Cancellation',
  cancellationSubtitle: 'Ctrl+C or API cancel stops the job, frees resources, and leaves no orphaned processes.',
  cancellationSteps: [
    {
      timestamp: 't+0ms',
      title: (
        <>
          Client sends <code className="bg-muted px-1 rounded text-xs">POST /v1/cancel</code>
        </>
      ),
      description: 'Idempotent request.',
    },
    {
      timestamp: 't+50ms',
      title: 'SSE disconnect detected',
      description: 'Stream closes ≤ 1s.',
    },
    {
      timestamp: 't+80ms',
      title: 'Immediate cleanup',
      description: 'Stop tokens, release slot, log event.',
    },
    {
      timestamp: 't+120ms',
      title: <span className="text-chart-3">Worker idle ✓</span>,
      description: 'Ready for next task.',
      variant: 'success',
    },
  ],
}

// === Security & Isolation ===

/**
 * Security & Isolation container - Layout configuration
 */
export const securityIsolationContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Security & Isolation',
  description: 'Defense-in-depth with six focused Rust crates. Enterprise-grade security for your homelab.',
  background: {

    variant: 'background',

  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Security & Isolation - Defense-in-depth architecture
 */
export const securityIsolationProps: SecurityIsolationProps = {
  cratesTitle: 'Six Specialized Security Crates',
  cratesSubtitle: 'Each concern ships as its own Rust crate—focused responsibility, no monolith.',
  securityCrates: [
    {
      name: 'auth-min',
      description: 'Timing-safe tokens, zero-trust auth.',
      hoverColor: 'hover:border-chart-2/50',
    },
    {
      name: 'audit-logging',
      description: 'Append-only logs, 7-year retention.',
      hoverColor: 'hover:border-chart-3/50',
    },
    {
      name: 'input-validation',
      description: 'Injection prevention, schema validation.',
      hoverColor: 'hover:border-primary/50',
    },
    {
      name: 'secrets-management',
      description: 'Encrypted storage, rotation, KMS-friendly.',
      hoverColor: 'hover:border-amber-500/50',
    },
    {
      name: 'jwt-guardian',
      description: 'RS256 validation, revocation lists, short-lived tokens.',
      hoverColor: 'hover:border-chart-2/50',
    },
    {
      name: 'deadline-propagation',
      description: 'Timeouts, cleanup, cascading shutdown.',
      hoverColor: 'hover:border-chart-3/50',
    },
  ],
  processIsolationTitle: 'Process Isolation',
  processIsolationSubtitle: 'Workers run in isolated processes with clean shutdown.',
  processFeatures: [
    { title: 'Sandboxed execution', color: 'chart-3' },
    { title: 'Cascading shutdown', color: 'chart-3' },
    { title: 'VRAM cleanup', color: 'chart-3' },
  ],
  zeroTrustTitle: 'Zero-Trust Architecture',
  zeroTrustSubtitle: 'Defense-in-depth with timing-safe auth and audit logs.',
  zeroTrustFeatures: [
    { title: 'Timing-safe authentication', color: 'chart-2' },
    { title: 'Immutable audit logs', color: 'chart-2' },
    { title: 'Input validation', color: 'chart-2' },
  ],
}

// === Additional Features Grid ===

/**
 * Additional Features Grid container - Layout configuration
 */
export const additionalFeaturesGridContainerProps: Omit<TemplateContainerProps, 'children'> = {
  eyebrow: <Badge variant="secondary">Capabilities overview</Badge>,
  title: 'Everything You Need for AI Infrastructure',
  background: {

    variant: 'background',

  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Additional Features Grid - Capabilities overview
 */
export const additionalFeaturesGridProps: AdditionalFeaturesGridProps = {
  rows: [
    {
      categoryLabel: 'Core Platform',
      cards: [
        {
          href: '#security-isolation',
          ariaLabel: 'Learn more about Cascading Shutdown',
          icon: <Shield className="size-6" />,
          iconTone: 'chart-2',
          title: 'Cascading Shutdown',
          subtitle: 'Ctrl+C tears down keeper → queen → hive → workers. No orphans, no VRAM leaks.',
          borderColor: 'before:h-[2px] before:bg-chart-2',
        },
        {
          href: '#intelligent-model-management',
          ariaLabel: 'Learn more about Model Catalog',
          icon: <Database className="size-6" />,
          iconTone: 'chart-3',
          title: 'Model Catalog',
          subtitle: 'Auto-provision models from Hugging Face with checksum verify and local cache.',
          borderColor: 'before:h-[2px] before:bg-chart-3',
        },
        {
          href: '#cross-node-orchestration',
          ariaLabel: 'Learn more about Network Orchestration',
          icon: <Network className="size-6" />,
          iconTone: 'primary',
          title: 'Network Orchestration',
          subtitle: 'Run jobs across gaming PCs, workstations, and Macs as one homelab cluster.',
          borderColor:
            'before:h-1.5 before:bg-gradient-to-r before:from-primary before:via-chart-3 before:to-amber-500',
          featured: true,
        },
      ],
    },
    {
      categoryLabel: 'Developer Tools',
      cards: [
        {
          href: '#cli-ui',
          ariaLabel: 'Learn more about CLI & Web UI',
          icon: <Terminal className="size-6" />,
          iconTone: 'muted',
          title: 'CLI & Web UI',
          subtitle: 'Automate with a fast CLI or manage visually in the web UI—your call.',
          borderColor: 'before:h-[2px] before:bg-muted-foreground',
        },
        {
          href: '#sdk',
          ariaLabel: 'Learn more about TypeScript SDK',
          icon: <Code className="size-6" />,
          iconTone: 'primary',
          title: 'TypeScript SDK',
          subtitle: 'Type-safe utilities for building agents; async/await with full IDE help.',
          borderColor: 'before:h-[2px] before:bg-primary',
        },
        {
          href: '#security-isolation',
          ariaLabel: 'Learn more about Security First',
          icon: <Shield className="size-6" />,
          iconTone: 'chart-2',
          title: 'Security First',
          subtitle: 'Six Rust crates: auth, audit logs, input validation, secrets, JWT guardian, and deadlines.',
          borderColor: 'before:h-[2px] before:bg-chart-2',
        },
      ],
    },
  ],
}

// === Email Capture ===

/**
 * Email capture content - Newsletter signup for features page
 */
export const featuresEmailCaptureProps: EmailCaptureProps = {
  headline: 'Stay updated on rbee development',
  subheadline: 'Get notified about new features, performance improvements, and community highlights.',
  emailInput: {
    placeholder: 'your@email.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Join Waitlist',
  },
  trustMessage: 'No spam. Unsubscribe anytime.',
  successMessage: "Thanks! You're on the list — we'll keep you posted.",
  communityFooter: {
    text: 'Follow progress & contribute on GitHub',
    linkText: 'View Repository',
    linkHref: 'https://github.com/veighnsche/llama-orch',
    subtext: 'Weekly dev notes. Roadmap issues tagged M0–M2.',
  },
  showBeeGlyphs: true,
  showIllustration: true,
}

/**
 * Email capture container - Background wrapper
 */
export const featuresEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}

/**
 * Features tabs container - Background wrapper
 */
export const featuresFeaturesTabsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
}
