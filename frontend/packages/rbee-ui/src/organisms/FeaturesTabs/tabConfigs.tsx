import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { GPUUtilizationBar } from '@rbee/ui/molecules/GPUUtilizationBar'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import { Code, Cpu, Gauge, Zap } from 'lucide-react'
import type { CoreFeaturesTabsProps } from './CoreFeaturesTabs'

export const defaultTabConfigs: CoreFeaturesTabsProps['tabs'] = [
  {
    value: 'api',
    icon: Code,
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
    benefits: [
      { text: 'No vendor lock-in' },
      { text: 'Use your models + GPUs' },
      { text: 'Keep existing tooling' },
    ],
  },
  {
    value: 'gpu',
    icon: Cpu,
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
    icon: Gauge,
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
    benefits: [
      { text: 'Deterministic routing' },
      { text: 'Policy & compliance ready' },
      { text: 'Easy to evolve' },
    ],
  },
  {
    value: 'sse',
    icon: Zap,
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
]
