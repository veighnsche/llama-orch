'use client'

import { NetworkMesh } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  CrossNodeOrchestrationProps,
  CTATemplateProps,
  EmailCaptureProps,
  FAQTemplateProps,
  HeroTemplateProps,
  HowItWorksProps,
  MultiBackendGpuTemplateProps,
  ProblemTemplateProps,
  ProvidersEarningsGPUModel,
  ProvidersEarningsPreset,
  ProvidersEarningsProps,
  SecurityIsolationProps,
  SolutionTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Box,
  Check,
  CheckCircle,
  Cpu,
  DollarSign,
  Download,
  GitBranch,
  HardDrive,
  Home as HomeIcon,
  Laptop,
  Lock,
  Monitor,
  Network,
  Power,
  Server,
  Settings,
  Shield,
  Terminal,
  Wifi,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Hero Section ===

/**
 * Hero section props - Homelab-focused hero with network topology visualization
 */
export const homelabHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'pulse',
    text: '100% Self-Hosted • Zero Cloud Dependencies',
  },
  headline: {
    variant: 'two-line-highlight',
    prefix: 'Your Homelab.',
    highlight: 'Your AI.',
  },
  subcopy:
    'Turn idle hardware into productive AI infrastructure. Run LLMs across all your machines—gaming PCs, old workstations, Mac minis. Complete privacy, zero cloud costs, SSH-first control.',
  proofElements: {
    variant: 'bullets',
    items: [
      { title: 'Use all your hardware (CUDA, Metal, CPU)', variant: 'check', color: 'chart-3' },
      { title: 'SSH-based orchestration', variant: 'check', color: 'chart-3' },
      { title: 'No telemetry, no tracking', variant: 'check', color: 'chart-3' },
    ],
  },
  ctas: {
    primary: {
      label: 'Download rbee',
      href: '/download',
      showIcon: true,
      dataUmamiEvent: 'cta:homelab-download',
    },
    secondary: {
      label: 'Setup Guide',
      href: '/docs/homelab-setup',
      variant: 'outline',
    },
  },
  trustElements: {
    variant: 'text',
    text: 'Compatible with Ubuntu, Debian, Arch, macOS. ARM64 and x86_64 supported.',
  },
  aside: (
    <div className="absolute inset-0 opacity-20">
      <NetworkMesh className="blur-[0.5px]" />
    </div>
  ),
  asideAriaLabel: 'Network topology visualization showing distributed homelab nodes',
  background: {
    variant: 'honeycomb',
    size: 'small',
    fadeDirection: 'radial',
  },
}

export const homelabHeroContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Email Capture Section ===

/**
 * Email capture for homelab setup guide
 */
export const homelabEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Free Setup Guide',
    showPulse: true,
  },
  headline: 'Get the Homelab Setup Guide',
  subheadline: 'Step-by-step instructions for setting up rbee across your homelab. Includes SSH configuration, GPU detection, and troubleshooting tips.',
  emailInput: {
    placeholder: 'your@email.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Send Me the Guide',
  },
  trustMessage: 'We respect your privacy. Unsubscribe anytime.',
  successMessage: 'Check your email! Setup guide sent.',
}

export const homelabEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '5xl',
}

// === Problem Section ===

/**
 * Problem section - Homelab complexity pain points
 */
export const homelabProblemProps: ProblemTemplateProps = {
  items: [
    {
      icon: <Server className="h-6 w-6" />,
      title: 'Scattered GPUs',
      body: 'Gaming PC in bedroom, workstation in office, old server in closet. All idle most of the time.',
    },
    {
      icon: <Settings className="h-6 w-6" />,
      title: 'Manual Setup Hell',
      body: 'Different OS versions, different CUDA drivers, different Python environments. Nothing works together.',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'No Orchestration',
      body: 'Want to run a 70B model? Good luck manually splitting it across machines and keeping track of processes.',
    },
    {
      icon: <DollarSign className="h-6 w-6" />,
      title: 'Wasted Hardware',
      body: 'Thousands of dollars in GPUs sitting idle because there\'s no easy way to use them together.',
    },
  ],
}

export const homelabProblemContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '6xl',
  background: {
    decoration: (
      <div className="absolute inset-0 opacity-5">
        <NetworkMesh />
      </div>
    ),
  },
}

// === Solution Section ===

/**
 * Solution section - Unified orchestration features
 */
export const homelabSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Network className="h-6 w-6" />,
      title: 'Unified Orchestration',
      body: 'One command to deploy models across all your machines. rbee handles SSH, GPU detection, and load balancing.',
    },
    {
      icon: <Terminal className="h-6 w-6" />,
      title: 'SSH-First Lifecycle',
      body: 'No agents, no daemons. Pure SSH control. Start, stop, monitor—all from your terminal.',
    },
    {
      icon: <Cpu className="h-6 w-6" />,
      title: 'Use All Your Hardware',
      body: 'CUDA, Metal, CPU—rbee detects and uses whatever you have. Mix and match architectures freely.',
    },
    {
      icon: <Lock className="h-6 w-6" />,
      title: 'Complete Privacy',
      body: 'Zero telemetry. Zero tracking. Your data never leaves your network. Open-source and auditable.',
    },
  ],
}

export const homelabSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '6xl',
}

// === How It Works Section ===

/**
 * How It Works - Step-by-step homelab setup
 */
export const homelabHowItWorksProps: HowItWorksProps = {
  steps: [
    {
      label: 'Install rbee-keeper',
      number: 1,
      block: {
        kind: 'code',
        title: 'Install on your main machine',
        language: 'bash',
        code: `# Ubuntu/Debian
curl -fsSL https://rbee.sh/install.sh | bash

# macOS
brew install rbee/tap/rbee-keeper

# Verify installation
rbee-keeper --version`,
      },
    },
    {
      label: 'Add Your Machines',
      number: 2,
      block: {
        kind: 'code',
        title: 'Register remote machines via SSH',
        language: 'bash',
        code: `# Add a machine to your pool
rbee-keeper pool add \\
  --name gaming-pc \\
  --host 192.168.1.100 \\
  --ssh-key ~/.ssh/id_ed25519

# Add multiple machines
rbee-keeper pool add --name workstation --host 192.168.1.101
rbee-keeper pool add --name mac-mini --host 192.168.1.102

# List all machines
rbee-keeper pool list`,
      },
    },
    {
      label: 'Deploy a Model',
      number: 3,
      block: {
        kind: 'code',
        title: 'Run models across your homelab',
        language: 'bash',
        code: `# Deploy a model across available GPUs
rbee-keeper infer \\
  --model llama-3.1-70b \\
  --pool gaming-pc,workstation \\
  --auto-balance

# rbee automatically:
# - Downloads the model if needed
# - Detects available GPUs
# - Splits model across machines
# - Starts inference server`,
      },
    },
    {
      label: 'Monitor & Manage',
      number: 4,
      block: {
        kind: 'code',
        title: 'Monitor your homelab',
        language: 'bash',
        code: `# Check status
rbee-keeper status

# View GPU utilization
rbee-keeper gpu-info --all

# Stop a model
rbee-keeper stop --model llama-3.1-70b

# Clean shutdown (no orphaned processes)
rbee-keeper shutdown --pool gaming-pc`,
      },
    },
  ],
}

export const homelabHowItWorksContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '6xl',
}

// === Cross-Node Orchestration Section ===

/**
 * Cross-Node Orchestration - Multi-machine visualization
 */
export const homelabCrossNodeProps: CrossNodeOrchestrationProps = {
  terminalContent: (
    <>
      <div className="text-muted-foreground">$ rbee-keeper pool list</div>
      <div className="mt-2 space-y-1">
        <div>
          <span className="text-chart-3">●</span> gaming-pc (192.168.1.100) - RTX 4090 24GB - READY
        </div>
        <div>
          <span className="text-chart-3">●</span> workstation (192.168.1.101) - RTX 3090 24GB - READY
        </div>
        <div>
          <span className="text-chart-3">●</span> mac-mini (192.168.1.102) - M2 Max 32GB - READY
        </div>
      </div>
      <div className="mt-4 text-muted-foreground">$ rbee-keeper infer --model llama-3.1-70b --auto-balance</div>
      <div className="mt-2 space-y-1">
        <div className="text-chart-2">→ Detected 3 machines with 72GB total VRAM</div>
        <div className="text-chart-2">→ Splitting model across gaming-pc + workstation</div>
        <div className="text-chart-3">✓ Model loaded in 12.3s</div>
        <div className="text-chart-3">✓ Inference server ready at http://localhost:8080</div>
      </div>
    </>
  ),
  terminalCopyText: `rbee-keeper pool list
rbee-keeper infer --model llama-3.1-70b --auto-balance`,
  benefits: [
    {
      icon: <CheckCircle className="h-5 w-5" />,
      title: 'Auto-Discovery',
      description: 'Detects GPUs, VRAM, and capabilities automatically',
    },
    {
      icon: <Zap className="h-5 w-5" />,
      title: 'Smart Balancing',
      description: 'Optimizes model placement across available hardware',
    },
    {
      icon: <Shield className="h-5 w-5" />,
      title: 'SSH Security',
      description: 'Uses your existing SSH keys and known_hosts',
    },
  ],
  diagramNodes: [
    { name: 'rbee-keeper', label: 'rbee-keeper\n(orchestrator)', tone: 'primary' },
    { name: 'gaming-pc', label: 'gaming-pc\nRTX 4090', tone: 'chart-2' },
    { name: 'workstation', label: 'workstation\nRTX 3090', tone: 'chart-3' },
  ],
  diagramArrows: [
    { label: 'SSH: Deploy model shard 1/2' },
    { label: 'SSH: Deploy model shard 2/2', indent: 'ml-16' },
  ],
  legendItems: [{ label: 'All communication via SSH' }, { label: 'No agents or daemons required' }],
  provisioningTitle: 'Automatic Provisioning',
  provisioningSubtitle: 'rbee-keeper orchestrates model deployment across your homelab via SSH',
}

export const homelabCrossNodeContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '7xl',
  background: {
    decoration: (
      <div className="absolute inset-0 opacity-5">
        <NetworkMesh />
      </div>
    ),
  },
}

// === Multi-Backend GPU Section ===

/**
 * Multi-Backend GPU - Hardware support matrix
 */
export const homelabMultiBackendProps: MultiBackendGpuTemplateProps = {
  policyTitle: 'Multi-Backend GPU Support',
  policySubtitle: 'CUDA, Metal, and CPU backends. rbee detects your hardware and lets you choose explicitly.',
  prohibitedBadges: [
    { label: 'No silent CPU fallback', variant: 'destructive' },
    { label: 'No hidden backend switches', variant: 'destructive' },
  ],
  whatHappensBadges: [
    { label: 'Explicit backend selection', variant: 'success' },
    { label: 'Clear hardware detection', variant: 'success' },
    { label: 'Helpful error messages', variant: 'success' },
  ],
  errorTitle: 'GPU not detected or insufficient VRAM',
  errorSuggestions: [
    'Check GPU drivers are installed (nvidia-smi for CUDA)',
    'Use --backend cpu to run on CPU explicitly',
    'Try a smaller quantized model (Q4_K_M)',
  ],
  terminalTitle: 'rbee detect — homelab-server.local',
  terminalContent: (
    <div className="space-y-1">
      <div className="text-chart-3">✓ CUDA detected: RTX 3080 (10GB VRAM)</div>
      <div className="text-chart-3">✓ CPU detected: 16 cores available</div>
      <div className="text-muted-foreground">✗ Metal: not available (Linux)</div>
    </div>
  ),
  backendDetections: [
    { label: 'cuda × 1', variant: 'primary' },
    { label: 'cpu × 1', variant: 'success' },
    { label: 'metal × 0', variant: 'muted' },
  ],
  totalDevices: 2,
  terminalFooter: 'Hardware detection cached for fast startup.',
  featureCards: [
    {
      icon: <Cpu className="size-6" />,
      title: 'NVIDIA CUDA',
      description: 'RTX 20/30/40 series, Tesla, A100, H100. Automatic detection and VRAM checks.',
    },
    {
      icon: <Laptop className="size-6" />,
      title: 'Apple Metal',
      description: 'M1, M2, M3 chips. Native Metal acceleration on Apple Silicon.',
    },
    {
      icon: <Server className="size-6" />,
      title: 'CPU Fallback',
      description: 'No GPU? Run on CPU with optimized quantization (Q4, Q8).',
    },
  ],
}

export const homelabMultiBackendContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '6xl',
}

// === Power Cost Calculator Section ===

/**
 * Power Cost Calculator - Adapted from ProvidersEarnings
 */
export const homelabPowerCostGPUModels: ProvidersEarningsGPUModel[] = [
  { name: 'RTX 4090', baseRate: 0.45, vram: 24 }, // 450W TDP
  { name: 'RTX 4080', baseRate: 0.32, vram: 16 }, // 320W TDP
  { name: 'RTX 3090', baseRate: 0.35, vram: 24 }, // 350W TDP
  { name: 'RTX 3080', baseRate: 0.32, vram: 10 }, // 320W TDP
  { name: 'RTX 3070', baseRate: 0.22, vram: 8 }, // 220W TDP
  { name: 'M2 Max', baseRate: 0.04, vram: 32 }, // 40W TDP
  { name: 'M1 Max', baseRate: 0.03, vram: 32 }, // 30W TDP
]

export const homelabPowerCostPresets: ProvidersEarningsPreset[] = [
  { label: 'Light Use (8h/day)', hours: 240, utilization: 50 },
  { label: 'Regular Use (12h/day)', hours: 360, utilization: 70 },
  { label: '24/7 Server', hours: 720, utilization: 80 },
]

export const homelabPowerCostProps: ProvidersEarningsProps = {
  gpuModels: homelabPowerCostGPUModels,
  presets: homelabPowerCostPresets,
  commission: 0, // No commission for power costs
  configTitle: 'Power Cost Calculator',
  selectGPULabel: 'Select Your GPU',
  presetsLabel: 'Usage Pattern',
  hoursLabel: 'Hours per Month',
  utilizationLabel: 'Average Load (%)',
  earningsTitle: 'Monthly Power Cost',
  monthlyLabel: 'Monthly',
  basedOnText: (hours: number, utilization: number) =>
    `Based on ${hours}h/month at ${utilization}% load, €0.30/kWh`,
  takeHomeLabel: 'Total Cost',
  dailyLabel: 'Daily',
  yearlyLabel: 'Yearly',
  breakdownTitle: 'Cost Breakdown',
  hourlyRateLabel: 'Power Draw (kW)',
  hoursPerMonthLabel: 'Hours per Month',
  utilizationBreakdownLabel: 'Average Load',
  commissionLabel: 'Electricity Rate',
  yourTakeHomeLabel: 'Your Monthly Cost',
  ctaLabel: 'Download rbee',
  ctaAriaLabel: 'Download rbee to start self-hosting',
  secondaryCTALabel: 'View Hardware Guide',
  formatCurrency: (n: number, opts?: Intl.NumberFormatOptions) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
      ...opts,
    }).format(n)
  },
  formatHourly: (n: number) => `${n.toFixed(2)} kW`,
}

export const homelabPowerCostContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '7xl',
}

// === Use Cases Section ===

/**
 * Use Cases - Different homelab setups
 */
export const homelabUseCasesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <HomeIcon className="h-6 w-6" />,
      title: 'Single PC Setup',
      scenario: 'One gaming PC with a single GPU. Want to run AI locally for personal use.',
      solution: 'Install rbee on your gaming PC. Run 7B-13B models locally with RTX 3070 or better.',
      outcome: 'Personal AI assistant, code completion, document processing. Zero cloud costs, complete privacy.',
      tags: ['Single GPU', 'Personal Use', 'Privacy'],
    },
    {
      icon: <Network className="h-6 w-6" />,
      title: 'Multi-Node Homelab',
      scenario: 'Multiple machines with GPUs scattered across your home. Want to run larger models.',
      solution: 'Connect gaming PC + workstation + old server with rbee. Automatic SSH orchestration.',
      outcome: 'Run 70B+ models by splitting across machines. Automatic load balancing, unified control.',
      tags: ['Multi-GPU', 'Large Models', 'Distributed'],
    },
    {
      icon: <Wifi className="h-6 w-6" />,
      title: 'Hybrid Setup',
      scenario: 'Need privacy for sensitive workloads but want cloud burst capacity.',
      solution: 'Mix local homelab GPUs with rented cloud GPUs. Privacy-first routing with rbee.',
      outcome: 'Best of both worlds. Local for privacy, cloud for burst. Cost-optimized workload distribution.',
      tags: ['Hybrid', 'Privacy', 'Cloud Burst'],
    },
  ],
  columns: 3,
}

export const homelabUseCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '6xl',
}

// === Security & Privacy Section ===

/**
 * Security & Privacy - Process isolation and network security
 */
export const homelabSecurityProps: SecurityIsolationProps = {
  cratesTitle: 'Security & Privacy First',
  cratesSubtitle: 'Built for homelabbers who care about security and privacy.',
  securityCrates: [
    {
      name: 'process-isolation',
      description: 'Each model runs in isolated process. Clean shutdown, no orphans.',
      hoverColor: 'hover:border-chart-3/50',
    },
    {
      name: 'ssh-security',
      description: 'Uses your existing SSH keys and known_hosts. No new attack surface.',
      hoverColor: 'hover:border-chart-2/50',
    },
    {
      name: 'zero-telemetry',
      description: 'No phone-home. No tracking. Your data stays on your network.',
      hoverColor: 'hover:border-primary/50',
    },
    {
      name: 'open-source',
      description: 'GPL-3.0-or-later. Fully auditable. No proprietary blobs.',
      hoverColor: 'hover:border-chart-2/50',
    },
  ],
  processIsolationTitle: 'Process Isolation',
  processIsolationSubtitle: 'Each worker runs in its own isolated process.',
  processFeatures: [
    { title: 'Clean shutdown on exit', color: 'chart-3' },
    { title: 'No orphaned processes', color: 'chart-3' },
    { title: 'VRAM cleanup guaranteed', color: 'chart-3' },
  ],
  zeroTrustTitle: 'Network Security',
  zeroTrustSubtitle: 'Your homelab, your rules. LAN-only mode available.',
  zeroTrustFeatures: [
    { title: 'LAN-only mode (no internet required)', color: 'chart-2' },
    { title: 'SSH-only communication (port 22)', color: 'chart-2' },
    { title: 'VPN compatible (WireGuard, Tailscale)', color: 'chart-2' },
  ],
}

export const homelabSecurityContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'lg',
  maxWidth: '6xl',
  background: {
    decoration: (
      <div className="absolute inset-0 opacity-5">
        <NetworkMesh />
      </div>
    ),
  },
}

// === FAQ Section ===

/**
 * FAQ - Homelab-specific questions
 */
export const homelabFAQProps: FAQTemplateProps = {
  badgeText: 'Homelab FAQ',
  categories: ['Hardware', 'Setup', 'Networking', 'Troubleshooting'],
  faqItems: [
    {
      value: 'hardware-requirements',
      question: 'What hardware do I need to run rbee?',
      answer:
        'Minimum: Any x86_64 or ARM64 machine with 8GB RAM. Recommended: Machine with NVIDIA GPU (RTX 20 series or newer) or Apple Silicon (M1/M2/M3). rbee works with CPU-only setups but GPU acceleration provides much better performance.',
      category: 'Hardware',
    },
    {
      value: 'mix-gpus',
      question: 'Can I mix different GPU types?',
      answer:
        'Yes! rbee supports mixing NVIDIA CUDA GPUs, Apple Metal GPUs, and CPU-only machines in the same pool. The orchestrator automatically detects capabilities and routes workloads appropriately.',
      category: 'Hardware',
    },
    {
      value: 'vram-requirements',
      question: 'How much VRAM do I need?',
      answer:
        'Depends on model size. 7B models: 6-8GB VRAM. 13B models: 12-16GB VRAM. 70B models: 48GB+ VRAM (can split across multiple GPUs). Use quantization (Q4, Q8) to reduce VRAM requirements.',
      category: 'Hardware',
    },
    {
      value: 'add-machine',
      question: 'How do I add a machine to my homelab pool?',
      answer:
        'Use `rbee-keeper pool add --name <name> --host <ip> --ssh-key <path>`. rbee will SSH into the machine, detect GPUs, and register it in your pool. Requires SSH access and sudo privileges on the remote machine.',
      category: 'Setup',
    },
    {
      value: 'remote-install',
      question: 'Do I need to install anything on remote machines?',
      answer:
        'rbee-keeper will automatically install the rbee-worker binary on remote machines via SSH. You need: SSH access, sudo privileges, and basic dependencies (curl, tar). No manual setup required.',
      category: 'Setup',
    },
    {
      value: 'docker-support',
      question: 'Can I use rbee with Docker/Podman?',
      answer:
        'Yes! rbee workers can run in containers. Use GPU passthrough (--gpus all for Docker, --device for Podman). See docs for container setup examples.',
      category: 'Setup',
    },
    {
      value: 'lan-only',
      question: 'Does rbee work on a LAN-only network?',
      answer:
        'Yes! After initial setup (downloading models), rbee works entirely offline. All communication is via SSH on your local network. No internet access required for inference.',
      category: 'Networking',
    },
    {
      value: 'vpn-support',
      question: 'Can I use rbee over a VPN?',
      answer:
        'Absolutely. rbee works seamlessly with WireGuard, Tailscale, OpenVPN, or any VPN solution. Just use the VPN IP addresses when adding machines to your pool.',
      category: 'Networking',
    },
    {
      value: 'ports',
      question: 'What ports does rbee use?',
      answer:
        'SSH (port 22) for orchestration. Inference server (default 8080, configurable). All communication is over standard protocols—no exotic ports required.',
      category: 'Networking',
    },
    {
      value: 'gpu-not-detected',
      question: 'My GPU is not detected. What should I do?',
      answer:
        'Check: 1) NVIDIA drivers installed (`nvidia-smi` works), 2) CUDA toolkit installed, 3) User has permissions to access GPU. Run `rbee-keeper gpu-info` for diagnostics. See troubleshooting guide for detailed steps.',
      category: 'Troubleshooting',
    },
    {
      value: 'manual-download',
      question: 'Model download is slow. Can I download manually?',
      answer:
        'Yes! Download models from Hugging Face manually and place in `~/.rbee/models/`. rbee will detect and use them. Use `rbee-keeper models import` to register manually downloaded models.',
      category: 'Troubleshooting',
    },
    {
      value: 'cleanup-processes',
      question: 'How do I clean up orphaned processes?',
      answer:
        'rbee tracks all processes and cleans up automatically on shutdown. Use `rbee-keeper cleanup` to force cleanup. Check `rbee-keeper status` to see running processes. No manual process hunting required.',
      category: 'Troubleshooting',
    },
  ],
}

export const homelabFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === CTA Section ===

/**
 * Final CTA - Download and get started
 */
export const homelabCTAProps: CTATemplateProps = {
  eyebrow: 'Ready to Get Started?',
  title: 'Turn Your Homelab Into an AI Powerhouse',
  subtitle: 'Download rbee and start running LLMs across all your machines. Free, open-source, and built for homelabbers.',
  primary: {
    label: 'Download rbee',
    href: '/download',
    iconLeft: Download,
  },
  secondary: {
    label: 'Read Setup Guide',
    href: '/docs/homelab-setup',
  },
  note: 'Works with your existing hardware • SSH-based orchestration • GPL-3.0-or-later',
  align: 'center',
  emphasis: 'gradient',
}

export const homelabCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  paddingY: 'xl',
  maxWidth: '5xl',
  background: {
    decoration: (
      <div className="absolute inset-0 opacity-10">
        <NetworkMesh />
      </div>
    ),
  },
}
