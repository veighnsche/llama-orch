// Part 2 of DevOpsPageProps - Error Handling, Real-Time Monitoring, Comparison, Compliance, FAQ, CTA

import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  ComparisonTemplateProps,
  EnterpriseComplianceProps,
  EnterpriseCTAProps,
  ErrorHandlingTemplateProps,
  FAQTemplateProps,
  RealTimeProgressProps,
} from '@rbee/ui/templates'
import { Activity, CheckCircle, Clock, Database, Lock, MessageSquare, Network, Rocket, Server, X } from 'lucide-react'

// === Error Handling Section ===

export const devopsErrorHandlingProps: ErrorHandlingTemplateProps = {
  statusKPIs: [
    {
      icon: <CheckCircle className="size-6" />,
      color: 'chart-3',
      label: 'Success Rate',
      value: '99.2%',
    },
    {
      icon: <Clock className="size-6" />,
      color: 'primary',
      label: 'Avg Recovery',
      value: '2.3s',
    },
    {
      icon: <Activity className="size-6" />,
      color: 'chart-2',
      label: 'Auto-Healed',
      value: '847',
    },
  ],
  terminalContent: (
    <>
      <div className="text-muted-foreground">
        <span className="text-chart-3">19:45:12</span> Worker gpu-01 health check failed
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-2">19:45:13</span> Retry 1/3 after 1s backoff
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-2">19:45:14</span> Retry 2/3 after 2s backoff
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-3">19:45:16</span> Worker gpu-01 recovered
      </div>
      <div className="text-muted-foreground">
        <span className="text-primary">19:45:17</span> Resuming normal operations
      </div>
    </>
  ),
  terminalFooter: (
    <span className="text-xs text-muted-foreground">
      Exponential backoff with jitter. Max 3 retries. Automatic failover after exhaustion.
    </span>
  ),
  playbookCategories: [
    {
      icon: <Network className="size-5" />,
      color: 'warning',
      title: 'Network Failures',
      checkCount: 5,
      severityDots: ['destructive', 'primary', 'chart-2', 'chart-3', 'chart-3'],
      description: 'SSH disconnects, timeouts, connection refused',
      checks: [
        {
          severity: 'destructive',
          title: 'SSH Connection Lost',
          meaning: 'Worker unreachable via SSH',
          actionLabel: 'Automatic Retry',
          href: '/docs/errors/ssh-lost',
          guideLabel: 'Recovery Guide',
          guideHref: '/docs/recovery/ssh',
        },
        {
          severity: 'primary',
          title: 'Health Check Timeout',
          meaning: 'Worker did not respond within 30s',
          actionLabel: 'Mark Unhealthy',
          href: '/docs/errors/health-timeout',
          guideLabel: 'Troubleshooting',
          guideHref: '/docs/troubleshooting/health',
        },
      ],
    },
    {
      icon: <Database className="size-5" />,
      color: 'primary',
      title: 'Resource Exhaustion',
      checkCount: 4,
      severityDots: ['destructive', 'destructive', 'primary', 'chart-2'],
      description: 'VRAM full, CPU overload, disk space',
      checks: [
        {
          severity: 'destructive',
          title: 'VRAM Exhausted',
          meaning: 'GPU memory full, cannot allocate',
          actionLabel: 'Reject Request',
          href: '/docs/errors/vram-full',
          guideLabel: 'Capacity Planning',
          guideHref: '/docs/capacity/vram',
        },
      ],
    },
    {
      icon: <Server className="size-5" />,
      color: 'chart-2',
      title: 'Process Failures',
      checkCount: 6,
      severityDots: ['destructive', 'destructive', 'primary', 'primary', 'chart-2', 'chart-3'],
      description: 'Worker crashes, daemon failures, orphaned processes',
      checks: [
        {
          severity: 'destructive',
          title: 'Worker Crashed',
          meaning: 'Worker process exited unexpectedly',
          actionLabel: 'Auto-Restart',
          href: '/docs/errors/worker-crash',
          guideLabel: 'Debug Guide',
          guideHref: '/docs/debugging/crashes',
        },
      ],
    },
  ],
}

export const devopsErrorHandlingContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Error Handling & Resilience',
  description: 'Automatic recovery with exponential backoff and proof bundles',
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Real-Time Monitoring Section ===

export const devopsRealTimeProgressProps: RealTimeProgressProps = {
  narrationTitle: 'Real-Time Worker Monitoring',
  narrationSubtitle: 'Live status updates via SSE streaming',
  terminalTitle: 'worker status stream',
  terminalAriaLabel: 'Real-time worker status stream',
  terminalContent: (
    <>
      <div className="text-muted-foreground">
        <span className="text-chart-3">●</span> gpu-worker-01: <span className="text-chart-3">ACTIVE</span> | VRAM:
        18.2/24GB | Latency: 42ms
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-3">●</span> gpu-worker-02: <span className="text-chart-3">ACTIVE</span> | VRAM:
        22.1/24GB | Latency: 38ms
      </div>
      <div className="text-muted-foreground">
        <span className="text-chart-2">●</span> gpu-worker-03: <span className="text-chart-2">STARTING</span> | VRAM:
        0.0/24GB | Latency: --
      </div>
      <div className="text-muted-foreground">
        <span className="text-destructive">●</span> gpu-worker-04: <span className="text-destructive">UNHEALTHY</span> |
        VRAM: 24.0/24GB | Latency: timeout
      </div>
    </>
  ),
  terminalFooter: (
    <span className="text-xs text-muted-foreground">
      SSE streaming. Updates every 5s. Automatic reconnect on disconnect.
    </span>
  ),
  metricKPIs: [
    {
      icon: <Activity className="size-6" />,
      color: 'chart-3',
      label: 'Active Workers',
      value: '12/15',
      progressPercentage: 80,
    },
    {
      icon: <Clock className="size-6" />,
      color: 'primary',
      label: 'Avg Latency',
      value: '45ms',
      progressPercentage: 65,
    },
    {
      icon: <Database className="size-6" />,
      color: 'chart-2',
      label: 'VRAM Usage',
      value: '68%',
      progressPercentage: 68,
    },
  ],
  cancellationTitle: 'Graceful Shutdown Sequence',
  cancellationSubtitle: 'Cascading stop prevents orphaned processes',
  cancellationSteps: [
    {
      timestamp: 'T+0s',
      title: <>Stop Daemon</>,
      description: 'Orchestrator daemon receives stop signal',
    },
    {
      timestamp: 'T+2s',
      title: <>Stop Hive</>,
      description: 'Hive manager stops accepting new tasks',
    },
    {
      timestamp: 'T+5s',
      title: <>Stop Workers</>,
      description: 'All workers receive shutdown signal',
    },
    {
      timestamp: 'T+8s',
      title: <>Cleanup Complete</>,
      description: 'All processes stopped, no orphans',
      variant: 'success',
    },
  ],
}

export const devopsRealTimeProgressContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Real-Time Monitoring',
  description: 'Live worker status, metrics streaming, and graceful shutdown',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Deployment Options Comparison ===

export const devopsComparisonProps: ComparisonTemplateProps = {
  columns: [
    { key: 'rbee', label: 'rbee', accent: true },
    { key: 'kubernetes', label: 'Kubernetes' },
    { key: 'manual', label: 'Manual SSH' },
  ],
  rows: [
    {
      feature: 'SSH Lifecycle Control',
      values: {
        rbee: true,
        kubernetes: 'Via kubectl',
        manual: true,
      },
    },
    {
      feature: 'Cascading Shutdown',
      values: {
        rbee: true,
        kubernetes: 'Pod lifecycle',
        manual: 'Manual',
      },
    },
    {
      feature: 'Health Monitoring',
      values: {
        rbee: '30s heartbeat',
        kubernetes: 'Liveness probes',
        manual: 'None',
      },
    },
    {
      feature: 'Metrics Export',
      values: {
        rbee: 'Prometheus',
        kubernetes: 'Prometheus',
        manual: 'None',
      },
    },
    {
      feature: 'Proof Bundles',
      values: {
        rbee: true,
        kubernetes: false,
        manual: false,
      },
    },
    {
      feature: 'Learning Curve',
      values: {
        rbee: 'Low',
        kubernetes: 'High',
        manual: 'Medium',
      },
    },
  ],
  legend: [
    {
      icon: <CheckCircle className="h-3.5 w-3.5 text-chart-3" />,
      label: 'Supported',
    },
    {
      icon: <X className="h-3.5 w-3.5 text-destructive" />,
      label: 'Not Available',
    },
  ],
  legendNote: 'Comparison as of October 2025',
}

export const devopsComparisonContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Deployment Options',
  description: 'Compare rbee with Kubernetes and manual SSH deployments',
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === SLAs & Guarantees Section ===

export const devopsComplianceProps: EnterpriseComplianceProps = {
  pillars: [
    {
      icon: <Activity className="size-8" />,
      title: 'Uptime Guarantee',
      subtitle: '99.9% availability target',
      titleId: 'uptime-guarantee',
      bullets: [
        { title: 'Automatic failover on worker failure' },
        { title: 'Health checks every 30 seconds' },
        { title: 'Graceful degradation under load' },
        { title: 'No single point of failure' },
      ],
      box: {
        heading: 'SLA Commitments',
        items: ['99.9% uptime', '< 100ms p99 latency', '< 5min recovery time'],
        checkmarkColor: 'chart-3',
      },
    },
    {
      icon: <Clock className="size-8" />,
      title: 'Response Time',
      subtitle: 'Low-latency inference',
      titleId: 'response-time',
      bullets: [
        { title: 'Sub-100ms p99 latency target' },
        { title: 'Automatic load balancing' },
        { title: 'Request queuing with priorities' },
        { title: 'Real-time metrics tracking' },
      ],
      box: {
        heading: 'Performance Targets',
        items: ['< 50ms p50', '< 100ms p99', '< 200ms p99.9'],
        checkmarkColor: 'primary',
      },
    },
    {
      icon: <MessageSquare className="size-8" />,
      title: 'Support & Docs',
      subtitle: 'Production support',
      titleId: 'support-docs',
      bullets: [
        { title: 'Comprehensive deployment docs' },
        { title: 'Operations playbook included' },
        { title: 'Community Discord support' },
        { title: 'Enterprise support available' },
      ],
      box: {
        heading: 'Support Channels',
        items: ['Documentation', 'Discord Community', 'Enterprise Support'],
        checkmarkColor: 'chart-2',
      },
    },
  ],
}

export const devopsComplianceContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'SLAs & Guarantees',
  description: 'Production-grade commitments for uptime, latency, and support',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === FAQ Section ===

export const devopsFAQProps: FAQTemplateProps = {
  badgeText: 'DevOps FAQ',
  categories: ['Deployment', 'Monitoring', 'Operations', 'Troubleshooting'],
  faqItems: [
    {
      value: 'q1',
      question: 'How does SSH-based deployment work?',
      answer:
        'rbee uses SSH to deploy and control workers across your infrastructure. The orchestrator daemon connects to worker nodes via SSH, spawns worker processes, and maintains control through the SSH session. This ensures clean lifecycle management and prevents orphaned processes.',
      category: 'Deployment',
    },
    {
      value: 'q2',
      question: 'What happens if a worker crashes?',
      answer:
        'Workers run in isolated processes. If a worker crashes, the orchestrator detects it via health checks (30s interval), marks it as unhealthy, and can automatically restart it or route traffic to healthy workers. The system continues operating with reduced capacity.',
      category: 'Operations',
    },
    {
      value: 'q3',
      question: 'How do I monitor worker health?',
      answer:
        'rbee provides real-time health monitoring via 30-second heartbeats. Workers report their status, VRAM usage, and latency. You can view this in the real-time console or export metrics to Prometheus for alerting.',
      category: 'Monitoring',
    },
    {
      value: 'q4',
      question: 'What is cascading shutdown?',
      answer:
        'Cascading shutdown is a clean lifecycle management pattern: stop daemon → stop hive → stop workers. This sequence ensures all processes are properly terminated and no orphaned workers remain consuming VRAM.',
      category: 'Operations',
    },
    {
      value: 'q5',
      question: 'Can I deploy to Kubernetes?',
      answer:
        'rbee is designed for SSH-based deployments. While you could run it in Kubernetes, you would lose some benefits like SSH lifecycle control. For Kubernetes, consider using native pod lifecycle management.',
      category: 'Deployment',
    },
    {
      value: 'q6',
      question: 'How do I debug worker failures?',
      answer:
        'rbee generates proof bundles for every failure. These bundles contain deterministic seeds, request/response logs, error stack traces, and timing information. This makes failures reproducible and debuggable.',
      category: 'Troubleshooting',
    },
    {
      value: 'q7',
      question: 'What metrics are exported?',
      answer:
        'rbee exports Prometheus-compatible metrics including worker status, latency histograms, VRAM utilization, request counts, and error rates. All metrics include labels for filtering and aggregation.',
      category: 'Monitoring',
    },
    {
      value: 'q8',
      question: 'How do I handle VRAM exhaustion?',
      answer:
        'When VRAM is exhausted, rbee rejects new requests with a clear error message. You can configure resource limits per worker and use automatic load balancing to distribute requests across workers with available VRAM.',
      category: 'Troubleshooting',
    },
  ],
  jsonLdEnabled: true,
}

export const devopsFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Frequently Asked Questions',
  description: 'Common questions about deployment, monitoring, and operations',
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === CTA Section ===

export const devopsCTAProps: EnterpriseCTAProps = {
  trustStats: [
    { value: '99.9%', label: 'Uptime SLA' },
    { value: '< 100ms', label: 'p99 Latency' },
    { value: '30s', label: 'Health Checks' },
    { value: 'Zero', label: 'Orphaned Workers' },
  ],
  ctaOptions: [
    {
      icon: <Rocket className="size-6" />,
      title: 'Deploy Now',
      body: 'Get started with SSH-based deployment in minutes',
      tone: 'primary',
      eyebrow: 'Quick Start',
      buttonText: 'View Deployment Docs',
      buttonHref: '/docs/deployment',
      buttonVariant: 'default',
      buttonAriaLabel: 'View deployment documentation',
    },
    {
      icon: <MessageSquare className="size-6" />,
      title: 'Join Community',
      body: 'Connect with DevOps teams using rbee in production',
      tone: 'outline',
      eyebrow: 'Get Support',
      buttonText: 'Discord Community',
      buttonHref: 'https://discord.gg/rbee',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Join Discord community',
    },
    {
      icon: <Lock className="size-6" />,
      title: 'Enterprise Support',
      body: 'Production SLAs, dedicated support, custom deployment',
      tone: 'outline',
      eyebrow: 'For Teams',
      buttonText: 'Contact Sales',
      buttonHref: '/enterprise/contact',
      buttonVariant: 'outline',
      buttonAriaLabel: 'Contact enterprise sales',
    },
  ],
}

export const devopsCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Ready to Deploy?',
  description: 'Start building production-ready AI infrastructure today',
  background: {
    variant: 'muted',
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}
