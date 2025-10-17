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

export const homelabHeroContainerProps: TemplateContainerProps = {
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
    icon: <Download className="h-3 w-3" />,
  },
  title: 'Get the Homelab Setup Guide',
  subtitle: 'Step-by-step instructions for setting up rbee across your homelab. Includes SSH configuration, GPU detection, and troubleshooting tips.',
  placeholder: 'your@email.com',
  buttonText: 'Send Me the Guide',
  privacyText: 'We respect your privacy. Unsubscribe anytime.',
  successMessage: 'Check your email! Setup guide sent.',
  errorMessage: 'Something went wrong. Please try again.',
}

export const homelabEmailCaptureContainerProps: TemplateContainerProps = {
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
      description: 'Gaming PC in bedroom, workstation in office, old server in closet. All idle most of the time.',
    },
    {
      icon: <Settings className="h-6 w-6" />,
      title: 'Manual Setup Hell',
      description: 'Different OS versions, different CUDA drivers, different Python environments. Nothing works together.',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'No Orchestration',
      description: 'Want to run a 70B model? Good luck manually splitting it across machines and keeping track of processes.',
    },
    {
      icon: <DollarSign className="h-6 w-6" />,
      title: 'Wasted Hardware',
      description: 'Thousands of dollars in GPUs sitting idle because there\'s no easy way to use them together.',
    },
  ],
}

export const homelabProblemContainerProps: TemplateContainerProps = {
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
      description: 'One command to deploy models across all your machines. rbee handles SSH, GPU detection, and load balancing.',
    },
    {
      icon: <Terminal className="h-6 w-6" />,
      title: 'SSH-First Lifecycle',
      description: 'No agents, no daemons. Pure SSH control. Start, stop, monitor—all from your terminal.',
    },
    {
      icon: <Cpu className="h-6 w-6" />,
      title: 'Use All Your Hardware',
      description: 'CUDA, Metal, CPU—rbee detects and uses whatever you have. Mix and match architectures freely.',
    },
    {
      icon: <Lock className="h-6 w-6" />,
      title: 'Complete Privacy',
      description: 'Zero telemetry. Zero tracking. Your data never leaves your network. Open-source and auditable.',
    },
  ],
}

export const homelabSolutionContainerProps: TemplateContainerProps = {
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
      content: {
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
      content: {
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
      content: {
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
      content: {
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

export const homelabHowItWorksContainerProps: TemplateContainerProps = {
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

export const homelabCrossNodeContainerProps: TemplateContainerProps = {
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
  policyTitle: 'Hardware Support',
  policySubtitle: 'rbee works with whatever hardware you have. Mix and match freely.',
  policies: [
    {
      icon: <Cpu className="h-6 w-6" />,
      title: 'NVIDIA CUDA',
      description: 'RTX 20/30/40 series, Tesla, A100, H100. Automatic CUDA detection and driver compatibility checks.',
    },
    {
      icon: <Laptop className="h-6 w-6" />,
      title: 'Apple Metal',
      description: 'M1, M2, M3 chips. Native Metal acceleration for maximum performance on Apple Silicon.',
    },
    {
      icon: <Server className="h-6 w-6" />,
      title: 'CPU Fallback',
      description: 'No GPU? No problem. rbee runs on CPU with optimized quantization (Q4, Q8).',
    },
  ],
  compatibilityTitle: 'Operating Systems',
  compatibilitySubtitle: 'Cross-platform support for all major homelab OSes',
  compatibilityItems: [
    {
      icon: <Monitor className="h-5 w-5" />,
      label: 'Ubuntu 20.04+',
      description: 'Full support',
    },
    {
      icon: <Monitor className="h-5 w-5" />,
      label: 'Debian 11+',
      description: 'Full support',
    },
    {
      icon: <Monitor className="h-5 w-5" />,
      label: 'Arch Linux',
      description: 'Full support',
    },
    {
      icon: <Laptop className="h-5 w-5" />,
      label: 'macOS 12+',
      description: 'Full support',
    },
    {
      icon: <Box className="h-5 w-5" />,
      label: 'NixOS',
      description: 'Community support',
    },
    {
      icon: <Server className="h-5 w-5" />,
      label: 'Proxmox',
      description: 'GPU passthrough supported',
    },
  ],
}

export const homelabMultiBackendContainerProps: TemplateContainerProps = {
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

export const homelabPowerCostContainerProps: TemplateContainerProps = {
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
      description:
        'One gaming PC with a single GPU. Perfect for personal AI assistant, code completion, and document processing. Start simple, scale later.',
      bullets: [
        'RTX 3070 or better',
        'Run 7B-13B models locally',
        'Zero cloud costs',
        'Complete privacy',
      ],
    },
    {
      icon: <Network className="h-6 w-6" />,
      title: 'Multi-Node Homelab',
      description:
        'Multiple machines with GPUs. Gaming PC + workstation + old server. Run larger models (70B+) by splitting across machines.',
      bullets: [
        '2-4 machines with GPUs',
        'Run 70B+ models',
        'Automatic load balancing',
        'SSH orchestration',
      ],
    },
    {
      icon: <Wifi className="h-6 w-6" />,
      title: 'Hybrid Setup',
      description:
        'Mix local and remote GPUs. Use your homelab for privacy-sensitive workloads, rent cloud GPUs for burst capacity. Best of both worlds.',
      bullets: [
        'Local + cloud GPUs',
        'Privacy-first routing',
        'Burst to cloud when needed',
        'Cost optimization',
      ],
    },
  ],
  columns: 3,
}

export const homelabUseCasesContainerProps: TemplateContainerProps = {
  paddingY: 'lg',
  maxWidth: '6xl',
}

// === Security & Privacy Section ===

/**
 * Security & Privacy - Process isolation and network security
 */
export const homelabSecurityProps: SecurityIsolationProps = {
  cratesTitle: 'Security & Privacy',
  cratesSubtitle: 'Built for homelabbers who care about security and privacy',
  securityCrates: [
    {
      name: 'Process Isolation',
      description: 'Each model runs in its own isolated process. Clean shutdown, no orphaned processes.',
      icon: <Box className="h-5 w-5" />,
      color: 'chart-3',
    },
    {
      name: 'SSH Security',
      description: 'Uses your existing SSH keys and known_hosts. No new attack surface.',
      icon: <Lock className="h-5 w-5" />,
      color: 'chart-2',
    },
    {
      name: 'Zero Telemetry',
      description: 'No phone-home. No tracking. No analytics. Your data stays on your network.',
      icon: <Shield className="h-5 w-5" />,
      color: 'chart-3',
    },
    {
      name: 'Open Source',
      description: 'GPL-3.0-or-later. Fully auditable. No proprietary blobs.',
      icon: <GitBranch className="h-5 w-5" />,
      color: 'chart-2',
    },
  ],
  isolationTitle: 'Network Security',
  isolationSubtitle: 'Your homelab, your rules',
  isolationFeatures: [
    {
      title: 'LAN-Only Mode',
      description: 'Run rbee entirely on your local network. No internet access required after initial setup.',
      icon: <Wifi className="h-5 w-5" />,
    },
    {
      title: 'Firewall Friendly',
      description: 'All communication via SSH (port 22). No exotic ports or protocols.',
      icon: <Shield className="h-5 w-5" />,
    },
    {
      title: 'VPN Compatible',
      description: 'Works seamlessly with WireGuard, Tailscale, or any VPN solution.',
      icon: <Network className="h-5 w-5" />,
    },
  ],
}

export const homelabSecurityContainerProps: TemplateContainerProps = {
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
  categories: [
    {
      id: 'hardware',
      label: 'Hardware',
      icon: <Cpu className="h-4 w-4" />,
    },
    {
      id: 'setup',
      label: 'Setup',
      icon: <Settings className="h-4 w-4" />,
    },
    {
      id: 'networking',
      label: 'Networking',
      icon: <Network className="h-4 w-4" />,
    },
    {
      id: 'troubleshooting',
      label: 'Troubleshooting',
      icon: <AlertTriangle className="h-4 w-4" />,
    },
  ],
  faqs: [
    {
      category: 'hardware',
      question: 'What hardware do I need to run rbee?',
      answer:
        'Minimum: Any x86_64 or ARM64 machine with 8GB RAM. Recommended: Machine with NVIDIA GPU (RTX 20 series or newer) or Apple Silicon (M1/M2/M3). rbee works with CPU-only setups but GPU acceleration provides much better performance.',
    },
    {
      category: 'hardware',
      question: 'Can I mix different GPU types?',
      answer:
        'Yes! rbee supports mixing NVIDIA CUDA GPUs, Apple Metal GPUs, and CPU-only machines in the same pool. The orchestrator automatically detects capabilities and routes workloads appropriately.',
    },
    {
      category: 'hardware',
      question: 'How much VRAM do I need?',
      answer:
        'Depends on model size. 7B models: 6-8GB VRAM. 13B models: 12-16GB VRAM. 70B models: 48GB+ VRAM (can split across multiple GPUs). Use quantization (Q4, Q8) to reduce VRAM requirements.',
    },
    {
      category: 'setup',
      question: 'How do I add a machine to my homelab pool?',
      answer:
        'Use `rbee-keeper pool add --name <name> --host <ip> --ssh-key <path>`. rbee will SSH into the machine, detect GPUs, and register it in your pool. Requires SSH access and sudo privileges on the remote machine.',
    },
    {
      category: 'setup',
      question: 'Do I need to install anything on remote machines?',
      answer:
        'rbee-keeper will automatically install the rbee-worker binary on remote machines via SSH. You need: SSH access, sudo privileges, and basic dependencies (curl, tar). No manual setup required.',
    },
    {
      category: 'setup',
      question: 'Can I use rbee with Docker/Podman?',
      answer:
        'Yes! rbee workers can run in containers. Use GPU passthrough (--gpus all for Docker, --device for Podman). See docs for container setup examples.',
    },
    {
      category: 'networking',
      question: 'Does rbee work on a LAN-only network?',
      answer:
        'Yes! After initial setup (downloading models), rbee works entirely offline. All communication is via SSH on your local network. No internet access required for inference.',
    },
    {
      category: 'networking',
      question: 'Can I use rbee over a VPN?',
      answer:
        'Absolutely. rbee works seamlessly with WireGuard, Tailscale, OpenVPN, or any VPN solution. Just use the VPN IP addresses when adding machines to your pool.',
    },
    {
      category: 'networking',
      question: 'What ports does rbee use?',
      answer:
        'SSH (port 22) for orchestration. Inference server (default 8080, configurable). All communication is over standard protocols—no exotic ports required.',
    },
    {
      category: 'troubleshooting',
      question: 'My GPU is not detected. What should I do?',
      answer:
        'Check: 1) NVIDIA drivers installed (`nvidia-smi` works), 2) CUDA toolkit installed, 3) User has permissions to access GPU. Run `rbee-keeper gpu-info` for diagnostics. See troubleshooting guide for detailed steps.',
    },
    {
      category: 'troubleshooting',
      question: 'Model download is slow. Can I download manually?',
      answer:
        'Yes! Download models from Hugging Face manually and place in `~/.rbee/models/`. rbee will detect and use them. Use `rbee-keeper models import` to register manually downloaded models.',
    },
    {
      category: 'troubleshooting',
      question: 'How do I clean up orphaned processes?',
      answer:
        'rbee tracks all processes and cleans up automatically on shutdown. Use `rbee-keeper cleanup` to force cleanup. Check `rbee-keeper status` to see running processes. No manual process hunting required.',
    },
  ],
}

export const homelabFAQContainerProps: TemplateContainerProps = {
  paddingY: 'xl',
  maxWidth: '5xl',
}

// === CTA Section ===

/**
 * Final CTA - Download and get started
 */
export const homelabCTAProps: CTATemplateProps = {
  eyebrow: 'Ready to Self-Host?',
  heading: 'Turn Your Homelab Into an AI Powerhouse',
  description:
    'Download rbee and start running LLMs across all your machines. Free, open-source, and built for homelabbers.',
  primaryCTA: {
    label: 'Download rbee',
    href: '/download',
    icon: <Download className="h-4 w-4" />,
  },
  secondaryCTA: {
    label: 'Read Setup Guide',
    href: '/docs/homelab-setup',
  },
  features: [
    {
      icon: <Check className="h-4 w-4" />,
      text: 'Works with your existing hardware',
    },
    {
      icon: <Check className="h-4 w-4" />,
      text: 'SSH-based orchestration',
    },
    {
      icon: <Check className="h-4 w-4" />,
      text: 'Zero cloud dependencies',
    },
    {
      icon: <Check className="h-4 w-4" />,
      text: 'GPL-3.0-or-later license',
    },
  ],
}

export const homelabCTAContainerProps: TemplateContainerProps = {
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
