'use client'

import { useCasesHero } from '@rbee/ui/assets'
import { IndustriesHero, UsecasesGridDark } from '@rbee/ui/icons'
import type { IndustryCardProps, TemplateContainerProps } from '@rbee/ui/molecules'
import type {
  EmailCaptureProps,
  UseCaseIndustryFilterItem,
  UseCasePrimaryFilterItem,
  UseCasePrimaryItem,
  UseCasesHeroTemplateProps,
  UseCasesIndustryTemplateProps,
  UseCasesPrimaryTemplateProps,
} from '@rbee/ui/templates'
import {
  Banknote,
  Briefcase,
  Building,
  Code,
  Factory,
  GraduationCap,
  Heart,
  Home,
  Landmark,
  Laptop,
  Scale,
  Server,
  Users,
} from 'lucide-react'

// ============================================================================
// Props Objects
// ============================================================================
// All props for the Use Cases page in visual order
// ============================================================================

// === UseCasesHero Template ===

/** Hero section - OpenAI-compatible independence message with homelab visual */
export const useCasesHeroProps: UseCasesHeroTemplateProps = {
  badgeText: 'OpenAI-compatible',
  heading: 'Built for Those Who Value',
  headingHighlight: 'Independence',
  description:
    'Own your AI infrastructure. From solo developers to enterprises, rbee adapts to your needs without compromising power or flexibility.',
  primaryCta: {
    text: 'Explore use cases',
    href: '#use-cases',
  },
  secondaryCta: {
    text: 'See architecture',
    href: '#architecture',
  },
  proofIndicators: [
    { text: 'Self-hosted', hasDot: true },
    { text: 'OpenAI-compatible' },
    { text: 'CUDA · Metal · CPU' },
  ],
  image: useCasesHero,
  imageAlt:
    "cinematic photoreal illustration of intimate homelab desk at night, shot from slightly elevated angle looking down at workspace, FOREGROUND LEFT: black mechanical keyboard with subtle white LED backlighting slightly out of focus creating soft glow, wireless mouse beside it, MIDDLE GROUND CENTER-LEFT: 15-inch MacBook Pro or ThinkPad laptop open showing full-screen terminal window with bright emerald green #10b981 monospace text streaming live AI token generation output 'Generating... token 847/2048' visible, screen has soft blue-white glow illuminating surroundings, small yellow Post-it note stuck to top bezel of laptop screen with handwritten black ink text 'your models your rules' in casual script, MIDDLE GROUND RIGHT: compact desktop mini tower or small rack unit approximately 12 inches tall containing 2-3 NVIDIA RTX 4090 or 3090 graphics cards visible through black mesh front panel with hexagonal perforations, each GPU has warm amber LED strips #f59e0b glowing along the edges creating horizontal light bars, soft orange rim light from GPUs casting warm glow on right side of desk surface and wall behind, faint heat shimmer effect above the GPU unit, DESK SURFACE: dark walnut or black wood finish desk with subtle wood grain texture, clean and minimal with only essential items, soft amber and teal reflections on glossy surface from various light sources, RIGHT EDGE: white ceramic coffee mug with thin wisps of steam rising, small potted succulent plant in concrete pot, BACKGROUND: deep midnight navy blue wall #0f172a with subtle texture, upper left corner has warm brass or copper desk lamp with conical shade creating focused pool of warm yellow light on desk, background fades to soft bokeh with circular light orbs in cool blue and warm amber tones suggesting depth, subtle teal accent light strip along wall edge, LIGHTING: key light from laptop screen (cool blue-white), fill light from desk lamp (warm yellow), accent light from GPUs (warm amber), rim light on hardware edges, subtle ambient glow from background, MOOD: cozy yet powerful, intimate workspace meets serious compute, warm inviting atmosphere with technical capability, sense of ownership and control, quiet confidence, STYLE: cinematic realism, shallow depth of field (f/2.8), soft focus on foreground keyboard, sharp focus on laptop screen and GPU unit, natural bokeh in background, color grading with warm shadows and cool highlights, subtle film grain, professional product photography aesthetic mixed with environmental portrait lighting",
  imageCaption: 'Your models, your hardware — no lock-in.',
}

// === UseCasesPrimary Template ===

/** Primary use cases container - wraps the use cases grid */
export const useCasesPrimaryContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Use Cases That Pay Off',
  bgVariant: 'background',
}

/** Primary use cases data - filters and use case cards */
const useCasesPrimaryFilters: UseCasePrimaryFilterItem[] = [
  { label: 'All', anchor: '#use-cases' },
  { label: 'Solo', anchor: '#developers' },
  { label: 'Team', anchor: '#use-cases' },
  { label: 'Enterprise', anchor: '#enterprise' },
  { label: 'Research', anchor: '#use-cases' },
]

const useCasesPrimaryItems: UseCasePrimaryItem[] = [
  {
    icon: <Laptop className="size-6" />,
    color: 'chart-2',
    title: 'Solo Developer',
    scenario:
      'Building a SaaS with AI assistance—needs Claude-level coding without creating a dependency on external providers. Worried about model changes, pricing shifts, or service shutdowns making the codebase unmaintainable.',
    solution:
      'Run rbee on gaming PC + spare workstation. Use Llama 70B for code generation, Stable Diffusion for assets. OpenAI-compatible API works with existing tools like Zed IDE.',
    outcome:
      'Complete independence. Models never change without permission. Code never leaves your network. Build AI coders that run forever on your hardware.',
    highlights: ['$0 inference', 'Private by default', 'No rate limits'],
    anchor: 'developers',
  },
  {
    icon: <Users className="size-6" />,
    color: 'primary',
    title: 'Small Team Startup',
    scenario:
      "5-person team burning ~$500/mo on AI APIs for code generation, docs, and support. Need to cut costs without sacrificing speed. Can't afford enterprise pricing but need professional-grade tools.",
    solution:
      'Pool 3 workstations + 2 Macs (8 GPUs total) into one rbee cluster. Multi-node orchestration distributes workload. SSH-based control plane keeps it simple.',
    outcome:
      'Save ~$6k/year while getting faster inference. Team owns the infrastructure. Scale by adding more hardware, not paying higher tiers.',
    highlights: ['Save ~$6k/yr', 'Faster tokens', 'GDPR-friendly'],
    badge: 'Most Popular',
  },
  {
    icon: <Home className="size-6" />,
    color: 'chart-3',
    title: 'Homelab Builder',
    scenario:
      'Four idle GPUs sitting unused—wants to build personal AI agents for coding, documentation, and automation. Tired of paying subscriptions for tools that could run locally.',
    solution:
      'Federate homelab hardware with rbee. Spin up specialized agents: AI coder for projects, docs generator for wikis, code reviewer for PRs. Use Rhai scripts to route tasks intelligently.',
    outcome:
      'Idle hardware becomes productive infrastructure. Build custom AI workflows without ongoing costs. Full control over models and routing logic.',
    highlights: ['Idle → productive', 'Zero ongoing fees', 'Full customization'],
    anchor: 'homelab',
  },
  {
    icon: <Building className="size-6" />,
    color: 'chart-4',
    title: 'Enterprise',
    scenario:
      '50-developer organization where code cannot leave the network due to compliance requirements. Need AI-assisted development but cloud APIs violate data residency policies. GDPR and audit trails are mandatory.',
    solution:
      'Deploy on-prem rbee with 20 GPUs. Rhai routing scripts enforce EU-only workers. Built-in audit logging tracks every API call. GDPR endpoints handle data export and deletion requests.',
    outcome:
      'Developers get AI assistance without compliance risk. Full audit trail for regulators. Data never leaves your infrastructure. EU-only routing guaranteed.',
    highlights: ['EU-only routing', 'Full audit trail', 'No external deps'],
    anchor: 'enterprise',
    badge: 'GDPR',
  },
  {
    icon: <Briefcase className="size-6" />,
    color: 'primary',
    title: 'Freelance Developer',
    scenario:
      'Works on multiple client projects under strict NDAs. Source code cannot touch external APIs or cloud services. Need AI coding assistance but client confidentiality is non-negotiable.',
    solution:
      'Run rbee entirely locally. All client code stays on the machine. Use Llama models for generation, refactoring, and documentation. OpenAI-compatible API works with favorite tools.',
    outcome:
      'Professional AI tools without breaking NDAs. Client code never leaves your laptop. No subscriptions, no usage tracking, complete confidentiality.',
    highlights: ['Client confidentiality', 'Pro-grade tools', 'Zero subscriptions'],
  },
  {
    icon: <GraduationCap className="size-6" />,
    color: 'chart-2',
    title: 'Research Lab',
    scenario:
      'Grant-funded university lab with tight cloud budget. Need reproducible experiments for papers. Cloud inference costs eat into research funding. Reproducibility requires deterministic outputs.',
    solution:
      'Deploy rbee on the lab GPU cluster. Spend grants on hardware (one-time cost) instead of cloud usage (recurring). Proof bundle system captures seeds and transcripts for reproducible runs.',
    outcome:
      'Maximize research budget by eliminating cloud fees. Reproducible experiments with deterministic testing. Own the infrastructure, control the models.',
    highlights: ['Maximize budget', 'Reproducible runs', 'No vendor lock-in'],
  },
  {
    icon: <Code className="size-6" />,
    color: 'chart-3',
    title: 'Open Source Maintainer',
    scenario:
      "Maintains popular OSS projects. Needs AI help with PR reviews, documentation generation, and issue triage. Can't afford enterprise API pricing. Want tools that align with open source values.",
    solution:
      'Run rbee on personal hardware. Build custom agents for PR review, docs generation, and issue triage. Use community-shared Rhai routing scripts. GPL license aligns with OSS philosophy.',
    outcome:
      'Sustainable AI tooling for OSS work. Help contributors without burning out. Community-aligned infrastructure that respects open source values.',
    highlights: ['Sustainable tooling', 'Community-aligned', 'No ongoing fees'],
  },
  {
    icon: <Server className="size-6" />,
    color: 'chart-4',
    title: 'GPU Provider',
    scenario:
      'Idle GPUs from former mining rigs or gaming PCs sitting unused. Want predictable monetization without complex setup. Interested in helping the community while earning passive income.',
    solution:
      'Join the rbee marketplace (M3 milestone). Set your own pricing and availability. Platform handles routing, billing, and SLA monitoring. Task-based pricing means you only earn when GPUs are used.',
    outcome:
      'Turn idle hardware into passive income. Help developers access affordable compute. Control your supply and pricing. Join a community-driven marketplace.',
    highlights: ['Passive income', 'Help the community', 'You control supply'],
  },
]

export const useCasesPrimaryProps: UseCasesPrimaryTemplateProps = {
  eyebrow: 'OpenAI-compatible • Your GPUs • Zero API Fees',
  heroImage: (
    <UsecasesGridDark size="100%" className="w-full h-auto" aria-label="Dark grid of LLM use cases: code, docs, chat" />
  ),
  heroImageAriaLabel: 'Dark grid of LLM use cases: code, docs, chat',
  filters: useCasesPrimaryFilters,
  useCases: useCasesPrimaryItems,
}

// === UseCasesIndustry Template ===

/** Industry use cases container - wraps the industry grid */
export const useCasesIndustryContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Industry-Specific Solutions',
  subtitle: 'rbee adapts to the unique compliance and security requirements of regulated industries.',
  bgVariant: 'secondary',
}

/** Industry use cases data - filters and industry cards */
const useCasesIndustryFilters: UseCaseIndustryFilterItem[] = [
  { label: 'All', anchor: '#architecture' },
  { label: 'Finance', anchor: '#finance' },
  { label: 'Healthcare', anchor: '#healthcare' },
  { label: 'Legal', anchor: '#legal' },
  { label: 'Public Sector', anchor: '#government' },
  { label: 'Education', anchor: '#education' },
  { label: 'Manufacturing', anchor: '#manufacturing' },
]

const useCasesIndustryItems: IndustryCardProps[] = [
  {
    title: 'Financial Services',
    icon: <Banknote className="size-6" />,
    color: 'primary',
    badge: 'GDPR',
    copy: 'GDPR-ready with audit trails and data residency. Run AI code review and risk analysis without sending financial data to external APIs.',
    anchor: 'finance',
  },
  {
    title: 'Healthcare',
    icon: <Heart className="size-6" />,
    color: 'chart-2',
    badge: 'HIPAA',
    copy: 'HIPAA-compliant by design. Patient data stays on your network while AI assists with medical coding, documentation, and research.',
    anchor: 'healthcare',
  },
  {
    title: 'Legal',
    icon: <Scale className="size-6" />,
    color: 'chart-3',
    copy: 'Preserve attorney–client privilege. Perform document/contract analysis and legal research with AI—without client data leaving your environment.',
    anchor: 'legal',
  },
  {
    title: 'Government',
    icon: <Landmark className="size-6" />,
    color: 'chart-4',
    badge: 'ITAR',
    copy: 'Sovereign, no foreign cloud dependency. Full auditability and policy-enforced routing to meet government security standards.',
    anchor: 'government',
  },
  {
    title: 'Education',
    icon: <GraduationCap className="size-6" />,
    color: 'chart-2',
    badge: 'FERPA',
    copy: 'Protect student information (FERPA-friendly). AI tutoring, grading assistance, and research tools with zero third-party data sharing.',
    anchor: 'education',
  },
  {
    title: 'Manufacturing',
    icon: <Factory className="size-6" />,
    color: 'primary',
    copy: 'Safeguard IP and trade secrets. AI-assisted CAD review, quality control, and process optimization—no exposure of proprietary designs.',
    anchor: 'manufacturing',
  },
]

export const useCasesIndustryProps: UseCasesIndustryTemplateProps = {
  eyebrow: 'Regulated sectors · Private-by-design',
  heroImage: (
    <IndustriesHero
      size="100%"
      className="w-full h-auto"
      aria-label="Visual representation of various industry sectors including healthcare, government, finance, education, manufacturing, and research with AI integration"
    />
  ),
  heroImageAriaLabel:
    'Visual representation of various industry sectors including healthcare, government, finance, education, manufacturing, and research with AI integration',
  filters: useCasesIndustryFilters,
  industries: useCasesIndustryItems,
}

// === EmailCapture Template ===

/** Email capture section - newsletter signup CTA */
export const useCasesEmailCaptureProps: EmailCaptureProps = {
  headline: 'Stay Updated',
  subheadline: 'Get notified about new features, use cases, and best practices.',
  emailInput: {
    placeholder: 'your@email.com',
    label: 'Email address',
  },
  submitButton: {
    label: 'Subscribe',
  },
  trustMessage: 'We respect your privacy. Unsubscribe at any time.',
  successMessage: 'Thanks for subscribing! Check your inbox.',
}

/**
 * Email capture container - Background wrapper
 */
export const useCasesEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}
