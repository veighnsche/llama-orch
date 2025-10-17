import { NetworkMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EmailCaptureProps,
  EnterpriseHeroProps,
  EnterpriseHowItWorksProps,
  EnterpriseSecurityProps,
  ErrorHandlingTemplateProps,
  ProblemTemplateProps,
  SolutionTemplateProps,
} from '@rbee/ui/templates'
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Database,
  FileText,
  GitBranch,
  Layers,
  Network,
  Rocket,
  Server,
  Settings,
  Shield,
  Terminal,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === DevOps Hero ===

export const devopsHeroProps: EnterpriseHeroProps = {
  badge: {
    icon: <Terminal className="size-6" />,
    text: 'Production-Ready Orchestration',
  },
  heading: 'Deploy AI Infrastructure That Actually Works',
  description:
    'SSH-first lifecycle. Cascading shutdown. No orphaned workers. Production-grade orchestration with health monitoring, graceful degradation, and clean lifecycle management.',
  stats: [
    {
      value: '30s',
      label: 'Heartbeat Interval',
      helpText: 'Automatic health checks every 30 seconds',
    },
    {
      value: 'Zero',
      label: 'Orphaned Workers',
      helpText: 'Cascading shutdown prevents orphaned processes',
    },
    {
      value: '99.9%',
      label: 'Uptime Target',
      helpText: 'Production SLA with automatic failover',
    },
  ],
  primaryCta: {
    text: 'View Deployment Docs',
    ariaLabel: 'View deployment documentation',
  },
  secondaryCta: {
    text: 'See Architecture',
    href: '#architecture',
  },
  helperText: 'SSH-based deployment. Multi-node orchestration. Clean lifecycle management.',
  complianceChips: [
    {
      icon: <Terminal className="h-3 w-3" />,
      label: 'SSH Control',
      ariaLabel: 'SSH-based lifecycle control',
    },
    {
      icon: <Activity className="h-3 w-3" />,
      label: 'Health Monitoring',
      ariaLabel: 'Automatic health monitoring',
    },
    {
      icon: <Shield className="h-3 w-3" />,
      label: 'Process Isolation',
      ariaLabel: 'Secure process isolation',
    },
  ],
  auditConsole: {
    title: 'Worker Lifecycle Console',
    badge: 'Live',
    filterButtons: [
      { label: 'All', ariaLabel: 'Filter: All workers', active: true },
      { label: 'Active', ariaLabel: 'Filter: Active workers' },
      { label: 'Deploying', ariaLabel: 'Filter: Deploying workers' },
      { label: 'Stopped', ariaLabel: 'Filter: Stopped workers' },
    ],
    events: [
      {
        event: 'worker.started',
        user: 'gpu-worker-01',
        time: '2025-10-17T19:45:23Z',
        displayTime: '2025-10-17 19:45:23 UTC',
        status: 'success',
      },
      {
        event: 'health.check',
        user: 'gpu-worker-02',
        time: '2025-10-17T19:45:15Z',
        displayTime: '2025-10-17 19:45:15 UTC',
        status: 'success',
      },
      {
        event: 'worker.shutdown',
        user: 'gpu-worker-03',
        time: '2025-10-17T19:44:58Z',
        displayTime: '2025-10-17 19:44:58 UTC',
        status: 'success',
      },
      {
        event: 'daemon.restart',
        user: 'orchestrator',
        time: '2025-10-17T19:44:32Z',
        displayTime: '2025-10-17 19:44:32 UTC',
        status: 'success',
      },
    ],
    footer: {
      retention: 'Real-time monitoring',
      tamperProof: 'SSH-controlled',
    },
  },
  floatingBadges: [
    {
      label: 'Active Workers',
      value: '12/15',
      ariaLabel: '12 of 15 workers active',
      position: 'top-right',
    },
    {
      label: 'Avg Response',
      value: '45ms',
      ariaLabel: 'Average response time 45 milliseconds',
      position: 'bottom-left',
    },
  ],
}

// === Email Capture ===

export const devopsEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'For DevOps Teams',
    showPulse: false,
  },
  headline: 'Production-ready AI orchestration',
  subheadline: 'Join DevOps teams that have eliminated orphaned workers and VRAM leaks.',
  emailInput: {
    placeholder: 'devops@company.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Get Deployment Guide',
  },
  trustMessage: 'SSH-first lifecycle. Clean shutdowns. No manual cleanup.',
  successMessage: 'Thanks! Your deployment guide is on the way.',
  communityFooter: {
    text: 'Read our deployment documentation',
    linkText: 'Deployment Docs',
    linkHref: '/docs/deployment',
    subtext: 'Download our operations playbook',
  },
}

export const devopsEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === Problem Section ===

export const devopsProblemProps: ProblemTemplateProps = {
  items: [
    {
      title: 'Orphaned Workers',
      body: 'Processes survive SSH disconnects. VRAM leaks accumulate. Manual cleanup required.',
      icon: <AlertTriangle className="h-6 w-6" />,
      tag: 'Production Risk',
      tone: 'destructive',
    },
    {
      title: 'No Observability',
      body: 'Black box deployments. No health checks. No metrics. Debug by SSH and grep.',
      icon: <Database className="h-6 w-6" />,
      tag: 'Blind Spots',
      tone: 'destructive',
    },
    {
      title: 'Manual Recovery',
      body: 'Worker crashes require manual intervention. No automatic failover. Downtime guaranteed.',
      icon: <Settings className="h-6 w-6" />,
      tag: 'Downtime',
      tone: 'destructive',
    },
  ],
}

export const devopsProblemContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Production Chaos',
  description: 'Why traditional AI deployments fail in production',
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Solution Section ===

export const devopsSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Terminal className="size-6" />,
      title: 'Cascading Shutdown',
      body: 'Daemon → Hive → Workers. Clean lifecycle prevents orphaned processes.',
    },
    {
      icon: <Activity className="size-6" />,
      title: 'Health Monitoring',
      body: '30-second heartbeats. Automatic detection of unhealthy workers.',
    },
    {
      icon: <Shield className="size-6" />,
      title: 'Process Isolation',
      body: 'Each worker in isolated process. Failures contained. No cascade.',
    },
    {
      icon: <Zap className="size-6" />,
      title: 'Graceful Degradation',
      body: 'Workers fail independently. System continues with reduced capacity.',
    },
  ],
  steps: [
    {
      title: 'SSH-First Lifecycle',
      body: 'Deploy via SSH. Control via SSH. Shutdown via SSH. No orphaned processes.',
    },
    {
      title: 'Automatic Health Checks',
      body: 'Heartbeats every 30 seconds. Automatic detection and recovery of failed workers.',
    },
    {
      title: 'Clean Shutdown Sequence',
      body: 'Stop daemon → Stop hive → Stop workers. Cascading shutdown guarantees cleanup.',
    },
  ],
}

export const devopsSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Production-Grade Orchestration',
  description: 'Built for reliability, observability, and clean lifecycle management',
  background: {
    variant: 'muted',
    decoration: (
      <div className="absolute inset-0 opacity-15">
        <NetworkMesh className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === How It Works Section ===

export const devopsHowItWorksProps: EnterpriseHowItWorksProps = {
  id: 'deployment-process',
  deploymentSteps: [
    {
      index: 1,
      icon: <Server className="size-6" />,
      title: 'Provision Infrastructure',
      intro: 'Set up your GPU nodes with SSH access',
      items: [
        'Configure SSH keys for remote access',
        'Install system dependencies (CUDA, drivers)',
        'Set up network topology and firewall rules',
        'Verify GPU availability and VRAM',
      ],
    },
    {
      index: 2,
      icon: <Settings className="size-6" />,
      title: 'Configure Orchestrator',
      intro: 'Define your deployment configuration',
      items: [
        'Specify worker endpoints and SSH credentials',
        'Configure health check intervals (default: 30s)',
        'Set resource limits and allocation policies',
        'Define failover and recovery strategies',
      ],
    },
    {
      index: 3,
      icon: <Rocket className="size-6" />,
      title: 'Deploy Workers',
      intro: 'SSH-based deployment to all nodes',
      items: [
        'Daemon starts on orchestrator node',
        'Hive spawns worker processes via SSH',
        'Workers register with orchestrator',
        'Health monitoring begins automatically',
      ],
    },
    {
      index: 4,
      icon: <Activity className="size-6" />,
      title: 'Monitor & Operate',
      intro: 'Real-time observability and control',
      items: [
        'View worker status in real-time console',
        'Monitor health checks and heartbeats',
        'Track metrics (latency, throughput, VRAM)',
        'Graceful shutdown via cascading stop',
      ],
    },
  ],
  timeline: {
    heading: 'Deployment Timeline',
    description: 'From zero to production in 4 weeks',
    weeks: [
      { week: 'Week 1', phase: 'Infrastructure Setup' },
      { week: 'Week 2', phase: 'Configuration & Testing' },
      { week: 'Week 3', phase: 'Staging Deployment' },
      { week: 'Week 4', phase: 'Production Rollout' },
    ],
  },
}

export const devopsHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Deployment Process',
  description: 'SSH-first lifecycle from provision to production',
  background: {
    variant: 'background',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Operational Features Section ===

export const devopsSecurityProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Terminal className="size-8" />,
      title: 'SSH Lifecycle Control',
      subtitle: 'Remote deployment and management',
      intro: 'Deploy and control workers via SSH. No manual server access required.',
      bullets: [
        'SSH-based worker deployment',
        'Remote daemon control',
        'Cascading shutdown sequence',
        'No orphaned processes',
      ],
      docsHref: '/docs/deployment/ssh',
    },
    {
      icon: <Activity className="size-8" />,
      title: 'Health Monitoring',
      subtitle: '30-second heartbeat checks',
      intro: 'Automatic detection of unhealthy workers with configurable intervals.',
      bullets: [
        'Heartbeat every 30 seconds',
        'Automatic failure detection',
        'Worker status tracking',
        'Real-time health dashboard',
      ],
      docsHref: '/docs/monitoring/health',
    },
    {
      icon: <FileText className="size-8" />,
      title: 'Structured Logging',
      subtitle: 'JSON logs for aggregation',
      intro: 'Machine-readable logs for easy parsing and aggregation.',
      bullets: [
        'JSON-formatted log output',
        'Structured event metadata',
        'Log aggregation ready',
        'Correlation IDs for tracing',
      ],
      docsHref: '/docs/observability/logging',
    },
    {
      icon: <Layers className="size-8" />,
      title: 'Metrics Emission',
      subtitle: 'Prometheus-compatible metrics',
      intro: 'Export metrics for monitoring and alerting systems.',
      bullets: ['Worker status metrics', 'Latency histograms', 'VRAM utilization tracking', 'Prometheus format'],
      docsHref: '/docs/observability/metrics',
    },
    {
      icon: <Shield className="size-8" />,
      title: 'Process Isolation',
      subtitle: 'Sandboxed worker processes',
      intro: 'Each worker runs in isolated process. Failures contained.',
      bullets: [
        'Isolated worker processes',
        'Resource limit enforcement',
        'Failure containment',
        'No cascade failures',
      ],
      docsHref: '/docs/architecture/isolation',
    },
    {
      icon: <GitBranch className="size-8" />,
      title: 'Proof Bundles',
      subtitle: 'Debugging artifacts',
      intro: 'Automatic capture of debugging artifacts for root cause analysis.',
      bullets: ['Deterministic seeds recorded', 'Request/response logs', 'Error stack traces', 'Reproducible failures'],
      docsHref: '/docs/debugging/proof-bundles',
    },
  ],
}

export const devopsSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Operational Features',
  description: 'Built for production reliability and observability',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
  maxWidth: '6xl',
}

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

// Re-export props from Part2 file
export {
  devopsComparisonContainerProps,
  devopsComparisonProps,
  devopsComplianceContainerProps,
  devopsComplianceProps,
  devopsCTAContainerProps,
  devopsCTAProps,
  devopsFAQContainerProps,
  devopsFAQProps,
  devopsRealTimeProgressContainerProps,
  devopsRealTimeProgressProps,
} from './DevOpsPageProps_Part2'
