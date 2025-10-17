import { Badge, NetworkMesh, OrchestrationFlow, StepFlow } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { TerminalWindow } from '@rbee/ui/molecules/TerminalWindow'
import type {
  AdditionalFeaturesGridProps,
  CTATemplateProps,
  EmailCaptureProps,
  FAQTemplateProps,
  FeaturesTabsProps,
  HeroTemplateProps,
  HowItWorksProps,
  ProblemTemplateProps,
  SolutionTemplateProps,
  TechnicalTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import {
  AlertTriangle,
  BookOpen,
  CheckCircle,
  Code2,
  FileCheck,
  FlaskConical,
  GitBranch,
  Image,
  Layers,
  Lock,
  Mic,
  RefreshCw,
  Rocket,
  Shield,
  Sparkles,
  Target,
  TestTube,
  Users,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Hero Section ===

/**
 * Hero section container - Layout configuration
 */
export const heroContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'gradient-primary',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-20 md:block">
        <NetworkMesh className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Hero section content - Reproducible AI for research
 */
export const heroProps: HeroTemplateProps = {
  badge: {
    variant: 'simple',
    text: 'Research-Grade AI',
  },
  headline: {
    variant: 'two-line-highlight',
    prefix: 'Reproducible AI for',
    highlight: 'Scientific Research',
  },
  subcopy:
    'Run reproducible experiments with deterministic seeds, proof bundles, and multi-modal support. Built for researchers who need verifiable results.',
  proofElements: {
    variant: 'bullets',
    items: [
      { title: 'Deterministic seeds', variant: 'check', color: 'chart-3' },
      { title: 'Proof bundles', variant: 'check', color: 'chart-3' },
      { title: 'Multi-modal support', variant: 'check', color: 'chart-3' },
    ],
  },
  ctas: {
    primary: {
      label: 'Explore Documentation',
      href: '/docs',
      showIcon: true,
    },
    secondary: {
      label: 'Join Waitlist',
      href: '#waitlist',
      variant: 'outline',
    },
  },
  aside: (
    <TerminalWindow
      title="experiment-runner.sh"
      showChrome={true}
      copyable={true}
      copyText={`rbee-keeper infer --model llama-3.1-70b --seed 42
→ Loading model (seed: 42)...
→ Model ready (2.1s)
→ Generating with deterministic seed...
→ Proof bundle: /experiments/exp_001/proof.json
→ Results: 100% reproducible ✓`}
    >
      <div className="space-y-2 font-mono text-sm">
        <div className="text-muted-foreground">$ rbee-keeper infer --model llama-3.1-70b --seed 42</div>
        <div className="text-chart-3">→ Loading model (seed: 42)...</div>
        <div className="text-chart-3">→ Model ready (2.1s)</div>
        <div>→ Generating with deterministic seed...</div>
        <div className="text-chart-2">→ Proof bundle: /experiments/exp_001/proof.json</div>
        <div className="text-chart-3">→ Results: 100% reproducible ✓</div>
      </div>
    </TerminalWindow>
  ),
}

// === Email Capture ===

/**
 * Email capture container
 */
export const emailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '3xl',
  align: 'center',
}

/**
 * Email capture content - Research-focused messaging
 */
export const emailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'For Researchers',
    showPulse: false,
  },
  headline: 'Join the Research Community',
  subheadline:
    'Get early access to reproducible AI infrastructure built for scientific computing and academic research.',
  emailInput: {
    placeholder: 'researcher@university.edu',
    label: 'Email address',
  },
  submitButton: {
    label: 'Join Research Waitlist',
  },
  trustMessage: 'No spam. Research updates only.',
  successMessage: "Thanks! We'll keep you updated on research features.",
  communityFooter: {
    text: 'Explore our research documentation',
    linkText: 'View Research Docs',
    linkHref: '/docs/research',
    subtext: 'Proof bundles, determinism suite, and experiment tracking.',
  },
}

// === Problem Template ===

/**
 * Problem template container
 */
export const problemContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Reproducibility Crisis in AI Research',
  description:
    "Research depends on reproducible results. But today's AI tools make reproducibility nearly impossible.",
  kickerVariant: 'destructive',
  background: {
    variant: 'gradient-destructive',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <NetworkMesh className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: 'xl',
  maxWidth: '7xl',
  align: 'center',
  ctaBanner: {
    copy: 'Build reproducible AI experiments with deterministic seeds, proof bundles, and verifiable results.',
    primary: { label: 'Explore Documentation', href: '/docs' },
    secondary: { label: 'View Architecture', href: '/docs/architecture' },
  },
}

/**
 * Problem template content - Research challenges
 */
export const problemProps: ProblemTemplateProps = {
  items: [
    {
      icon: <RefreshCw className="h-6 w-6" />,
      title: 'Non-Deterministic Results',
      body: 'Run the same experiment twice, get different results. No seed control. No reproducibility. Cannot verify findings.',
      tone: 'destructive',
      tag: 'Unreproducible',
    },
    {
      icon: <FileCheck className="h-6 w-6" />,
      title: 'Missing Proof Bundles',
      body: 'No audit trail of what ran. Cannot prove results. Reviewers cannot verify. Replication studies fail.',
      tone: 'destructive',
      tag: 'No verification',
    },
    {
      icon: <GitBranch className="h-6 w-6" />,
      title: 'Collaboration Barriers',
      body: 'Cannot share experiments reliably. Version mismatches. Environment differences. Results diverge across labs.',
      tone: 'destructive',
      tag: 'Fragmented',
    },
    {
      icon: <AlertTriangle className="h-6 w-6" />,
      title: 'Vendor Lock-In',
      body: 'Dependent on external APIs. Models change without notice. Cannot guarantee long-term reproducibility.',
      tone: 'destructive',
      tag: 'Unstable',
    },
  ],
  gridClassName: 'md:grid-cols-2 lg:grid-cols-4',
}

// === Solution Template ===

/**
 * Solution template container
 */
export const solutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Reproducible AI Infrastructure',
  description:
    'rbee provides deterministic execution, proof bundles, and experiment tracking—built for research that needs to be verifiable.',
  kicker: 'How rbee Works',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
  ctas: {
    primary: {
      label: 'Get Started',
      href: '/docs/quickstart',
      ariaLabel: 'Get started with rbee for research',
    },
    secondary: {
      label: 'View Examples',
      href: '/docs/examples',
      ariaLabel: 'View research examples',
    },
    caption: 'Open-source • GPL-3.0-or-later • Research-ready',
  },
}

/**
 * Solution template content - Reproducibility features
 */
export const solutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Target className="h-6 w-6" aria-hidden="true" />,
      title: 'Deterministic Seeds',
      body: 'Set seed once, get identical results every time. Full control.',
      badge: 'Reproducible',
    },
    {
      icon: <FileCheck className="h-6 w-6" aria-hidden="true" />,
      title: 'Proof Bundles',
      body: 'Immutable records of every experiment. Verifiable by reviewers.',
      badge: 'Auditable',
    },
    {
      icon: <GitBranch className="h-6 w-6" aria-hidden="true" />,
      title: 'Experiment Tracking',
      body: 'Version control for models, data, and results. Full lineage.',
    },
    {
      icon: <Lock className="h-6 w-6" aria-hidden="true" />,
      title: 'Model Versioning',
      body: 'Lock model versions. No surprise updates. Long-term stability.',
    },
  ],
  steps: [
    {
      title: 'Set Deterministic Seed',
      body: 'Configure seed for reproducible generation across all runs.',
    },
    {
      title: 'Run Experiment',
      body: 'Execute inference with proof bundle generation enabled.',
    },
    {
      title: 'Collect Proof Bundle',
      body: 'Immutable record includes seed, model version, inputs, outputs.',
    },
    {
      title: 'Verify & Share',
      body: 'Reviewers can verify results using the same seed and model version.',
    },
  ],
  earnings: {
    title: 'Reproducibility Metrics',
    rows: [
      {
        model: 'Determinism',
        meta: 'Seed-based',
        value: '100%',
        note: 'identical',
      },
      {
        model: 'Proof Bundles',
        meta: 'Immutable',
        value: 'Always',
        note: 'verifiable',
      },
      {
        model: 'Model Versions',
        meta: 'Locked',
        value: 'Stable',
        note: 'no drift',
      },
    ],
    disclaimer: 'Reproducibility guaranteed when using identical seed, model version, and configuration.',
  },
}

// === Multi-Modal Support ===

/**
 * Multi-modal support container
 */
export const multiModalContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
}

/**
 * Multi-modal support content - LLMs, Images, Audio, Embeddings
 */
export const multiModalProps: FeaturesTabsProps = {
  title: 'Multi-Modal AI Support',
  description: 'Run text, image, audio, and embedding models—all with deterministic seeds and proof bundles.',
  tabs: [
    {
      value: 'text',
      icon: <Code2 className="size-6" />,
      label: 'Text Generation (LLMs)',
      mobileLabel: 'Text',
      subtitle: 'Language models',
      badge: 'LLMs',
      description: 'Run Llama, Mistral, and other LLMs with deterministic seeds for reproducible text generation.',
      content: (
        <CodeBlock
          code={`rbee-keeper infer \\
  --model llama-3.1-70b \\
  --seed 42 \\
  --prompt "Explain quantum computing" \\
  --proof-bundle /experiments/exp_001/

→ Deterministic generation (seed: 42)
→ Proof bundle: exp_001/proof.json ✓`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Reproducible text generation with seed control.',
        variant: 'success',
      },
      benefits: [
        { text: 'Llama, Mistral, Qwen support' },
        { text: 'Deterministic seeds' },
        { text: 'Proof bundles included' },
      ],
    },
    {
      value: 'image',
      icon: <Image className="size-6" />,
      label: 'Image Generation',
      mobileLabel: 'Image',
      subtitle: 'Stable Diffusion',
      badge: 'SD',
      description: 'Generate images with Stable Diffusion using deterministic seeds for reproducible visual outputs.',
      content: (
        <CodeBlock
          code={`rbee-keeper infer \\
  --model stable-diffusion-xl \\
  --seed 42 \\
  --prompt "A scientific laboratory" \\
  --proof-bundle /experiments/img_001/

→ Image generation (seed: 42)
→ Output: img_001/output.png
→ Proof bundle: img_001/proof.json ✓`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Reproducible image generation with seed control.',
        variant: 'success',
      },
      benefits: [
        { text: 'Stable Diffusion XL support' },
        { text: 'Deterministic seeds' },
        { text: 'Proof bundles with metadata' },
      ],
    },
    {
      value: 'audio',
      icon: <Mic className="size-6" />,
      label: 'Text-to-Speech',
      mobileLabel: 'Audio',
      subtitle: 'TTS models',
      badge: 'TTS',
      description: 'Generate speech from text with deterministic seeds for reproducible audio outputs.',
      content: (
        <CodeBlock
          code={`rbee-keeper infer \\
  --model bark-tts \\
  --seed 42 \\
  --text "Hello from rbee" \\
  --proof-bundle /experiments/tts_001/

→ TTS generation (seed: 42)
→ Output: tts_001/output.wav
→ Proof bundle: tts_001/proof.json ✓`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Reproducible audio generation with seed control.',
        variant: 'success',
      },
      benefits: [{ text: 'Bark TTS support' }, { text: 'Deterministic seeds' }, { text: 'Proof bundles included' }],
    },
    {
      value: 'embeddings',
      icon: <Layers className="size-6" />,
      label: 'Embeddings',
      mobileLabel: 'Embed',
      subtitle: 'Vector embeddings',
      badge: 'Vectors',
      description: 'Generate embeddings for semantic search and similarity with deterministic results.',
      content: (
        <CodeBlock
          code={`rbee-keeper embed \\
  --model all-minilm-l6-v2 \\
  --text "Research paper abstract" \\
  --proof-bundle /experiments/emb_001/

→ Embedding generation
→ Vector: [0.123, -0.456, ...]
→ Proof bundle: emb_001/proof.json ✓`}
          language="bash"
          copyable={true}
        />
      ),
      highlight: {
        text: 'Reproducible embeddings for semantic search.',
        variant: 'success',
      },
      benefits: [
        { text: 'Sentence transformers support' },
        { text: 'Deterministic results' },
        { text: 'Proof bundles included' },
      ],
    },
  ],
  defaultTab: 'text',
  sectionId: 'multi-modal',
}

// === Research Workflow ===

/**
 * Research workflow container
 */
export const workflowContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Research Workflow',
  description: 'From experiment setup to result verification—streamlined for reproducible research.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Research workflow content - Step-by-step process
 */
export const workflowProps: HowItWorksProps = {
  steps: [
    {
      label: 'Setup Experiment',
      block: {
        kind: 'code',
        title: 'experiment.yaml',
        language: 'yaml',
        code: `experiment:
  name: "exp_001"
  model: "llama-3.1-70b"
  seed: 42
  proof_bundle: true`,
      },
    },
    {
      label: 'Run Inference',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>rbee-keeper infer --config exp_001.yaml</div>
            <div className="text-chart-3">→ Loading model (seed: 42)...</div>
            <div className="text-chart-3">→ Model ready (2.1s)</div>
            <div>→ Generating...</div>
            <div className="text-chart-2">→ Proof bundle: exp_001/proof.json ✓</div>
          </>
        ),
        copyText: `rbee-keeper infer --config exp_001.yaml
→ Loading model (seed: 42)...
→ Model ready (2.1s)
→ Generating...
→ Proof bundle: exp_001/proof.json ✓`,
      },
    },
    {
      label: 'Collect Results',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>ls exp_001/</div>
            <div className="text-muted-foreground">  output.txt</div>
            <div className="text-muted-foreground">  proof.json</div>
            <div className="text-muted-foreground">  metadata.json</div>
          </>
        ),
        copyText: `ls exp_001/
  output.txt
  proof.json
  metadata.json`,
      },
    },
    {
      label: 'Verify & Share',
      block: {
        kind: 'terminal',
        title: 'terminal',
        lines: (
          <>
            <div>rbee-keeper verify --proof exp_001/proof.json</div>
            <div className="text-chart-3">→ Verification: PASSED ✓</div>
            <div className="text-muted-foreground">→ Seed: 42</div>
            <div className="text-muted-foreground">→ Model: llama-3.1-70b (v1.0.0)</div>
            <div className="text-chart-3">→ Results: 100% reproducible</div>
          </>
        ),
        copyText: `rbee-keeper verify --proof exp_001/proof.json
→ Verification: PASSED ✓
→ Seed: 42
→ Model: llama-3.1-70b (v1.0.0)
→ Results: 100% reproducible`,
      },
    },
  ],
}

// === Determinism Suite ===

/**
 * Determinism suite container
 */
export const determinismContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Determinism Suite',
  description: 'Built-in tools for regression testing, seed management, and result verification.',
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
  align: 'center',
}

/**
 * Determinism suite content - Testing and verification tools
 */
export const determinismProps: AdditionalFeaturesGridProps = {
  rows: [
    {
      categoryLabel: 'Determinism Tools',
      cards: [
        {
          href: '/docs/determinism/regression',
          ariaLabel: 'Learn about regression testing',
          icon: <TestTube className="h-6 w-6" />,
          iconTone: 'chart-2',
          title: 'Regression Testing',
          subtitle: 'Automated tests to detect non-deterministic behavior',
          borderColor: 'border-chart-2/20',
        },
        {
          href: '/docs/determinism/seeds',
          ariaLabel: 'Learn about seed management',
          icon: <Target className="h-6 w-6" />,
          iconTone: 'chart-3',
          title: 'Seed Management',
          subtitle: 'Centralized seed registry for reproducibility',
          borderColor: 'border-chart-3/20',
        },
        {
          href: '/docs/determinism/verification',
          ariaLabel: 'Learn about result verification',
          icon: <CheckCircle className="h-6 w-6" />,
          iconTone: 'primary',
          title: 'Result Verification',
          subtitle: 'Verify proof bundles and detect tampering',
          borderColor: 'border-primary/20',
        },
        {
          href: '/docs/determinism/debugging',
          ariaLabel: 'Learn about debugging tools',
          icon: <FlaskConical className="h-6 w-6" />,
          iconTone: 'muted',
          title: 'Debugging Tools',
          subtitle: 'Trace non-deterministic sources',
          borderColor: 'border-muted/20',
        },
      ],
    },
  ],
}

// === Academic Use Cases ===

/**
 * Academic use cases container
 */
export const useCasesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Built for Academic Research',
  description: 'From research papers to thesis work—rbee supports reproducible AI research.',
  background: {
    variant: 'secondary',
  },
  paddingY: '2xl',
  maxWidth: '6xl',
  align: 'center',
}

/**
 * Academic use cases content - Research scenarios
 */
export const useCasesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <BookOpen className="h-6 w-6" />,
      title: 'Research Papers',
      scenario: 'Publishing AI research that needs to be reproducible by reviewers.',
      solution:
        'Use deterministic seeds and proof bundles. Reviewers can verify results using the same seed and model version.',
      outcome: 'Reproducible results. Faster peer review. Higher credibility.',
    },
    {
      icon: <FlaskConical className="h-6 w-6" />,
      title: 'Thesis Work',
      scenario: 'PhD student running long-term experiments that must be reproducible.',
      solution: 'Lock model versions, use deterministic seeds, and generate proof bundles for every experiment run.',
      outcome: 'Verifiable results. No model drift. Thesis committee approval.',
    },
    {
      icon: <Users className="h-6 w-6" />,
      title: 'Collaborative Projects',
      scenario: 'Multi-lab collaboration requiring consistent results across institutions.',
      solution:
        'Share experiment configs with seeds. Each lab runs identical experiments with verifiable proof bundles.',
      outcome: 'Consistent results. No environment mismatches. Faster collaboration.',
    },
    {
      icon: <Sparkles className="h-6 w-6" />,
      title: 'Teaching & Education',
      scenario: 'Teaching ML courses where students need reproducible assignments.',
      solution: 'Provide students with seed-based assignments. Verify submissions using proof bundles.',
      outcome: 'Fair grading. Reproducible assignments. No plagiarism concerns.',
    },
    {
      icon: <GitBranch className="h-6 w-6" />,
      title: 'Replication Studies',
      scenario: 'Replicating published research to verify findings.',
      solution: 'Use original seed and model version from proof bundle. Verify results match published findings.',
      outcome: 'Exact replication. Verify or refute claims. Advance science.',
    },
    {
      icon: <Rocket className="h-6 w-6" />,
      title: 'Grant Applications',
      scenario: 'Demonstrating reproducibility for grant proposals and funding.',
      solution: 'Include proof bundles in grant applications. Show verifiable, reproducible methodology.',
      outcome: 'Stronger proposals. Higher funding success. Credible research.',
    },
  ],
  columns: 3,
}

// === Technical Architecture ===

/**
 * Technical architecture container
 */
export const technicalContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '7xl',
}

/**
 * Technical architecture content - BDD-tested, Candle-powered
 */
export const technicalProps: TechnicalTemplateProps = {
  architectureHighlights: [
    {
      title: 'BDD-Tested',
      details: ['Executable specs ensure reproducibility guarantees'],
    },
    {
      title: 'Candle-Powered',
      details: ['Rust ML framework for memory efficiency and speed'],
    },
    {
      title: 'Deterministic',
      details: ['Seed-based generation for 100% reproducibility'],
    },
    {
      title: 'Proof Bundles',
      details: ['Immutable audit trails for every experiment'],
    },
  ],
  coverageProgress: {
    label: 'Test Coverage',
    passing: 87,
    total: 100,
  },
  techStack: [
    { name: 'Rust', description: 'Core orchestration', ariaLabel: 'Technology: Rust' },
    { name: 'Candle', description: 'ML framework', ariaLabel: 'Technology: Candle' },
    { name: 'BDD', description: 'Executable specs', ariaLabel: 'Technology: BDD' },
    { name: 'Proof Bundles', description: 'Audit trails', ariaLabel: 'Technology: Proof Bundles' },
  ],
  stackLinks: {
    githubUrl: 'https://github.com/veighnsche/llama-orch',
    license: 'GPL-3.0-or-later',
    architectureUrl: '/docs/architecture',
  },
}

// === FAQ ===

/**
 * FAQ container
 */
export const faqContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: '2xl',
  maxWidth: '5xl',
}

/**
 * FAQ content - Research-specific questions
 */
export const faqProps: FAQTemplateProps = {
  badgeText: 'FAQ',
  categories: ['General', 'Technical', 'Research'],
  faqItems: [
    {
      value: 'q1',
      question: 'How does rbee ensure reproducibility?',
      answer:
        'rbee uses deterministic seeds for all inference operations. When you set a seed, the same model with the same input will always produce identical outputs. Proof bundles record the seed, model version, inputs, and outputs for verification.',
      category: 'General',
    },
    {
      value: 'q2',
      question: 'What are proof bundles?',
      answer:
        'Proof bundles are immutable records of experiment runs. They include the seed, model version, configuration, inputs, outputs, and timestamps. Reviewers can use proof bundles to verify your results are reproducible.',
      category: 'General',
    },
    {
      value: 'q3',
      question: 'Can I use rbee for multi-modal research?',
      answer:
        'Yes! rbee supports LLMs (text), Stable Diffusion (images), TTS (audio), and embeddings. All modalities support deterministic seeds and proof bundles.',
      category: 'Technical',
    },
    {
      value: 'q4',
      question: 'How do I share experiments with collaborators?',
      answer:
        'Share your experiment config (including seed) and proof bundle. Collaborators can run the same experiment with identical results using the same seed and model version.',
      category: 'Research',
    },
    {
      value: 'q5',
      question: 'What models are supported?',
      answer:
        'rbee supports Llama, Mistral, Qwen (text), Stable Diffusion (images), Bark (TTS), and sentence transformers (embeddings). All models support deterministic seeds.',
      category: 'Technical',
    },
    {
      value: 'q6',
      question: 'How do I verify a proof bundle?',
      answer:
        'Use `rbee-keeper verify --proof <path>` to verify a proof bundle. This checks the seed, model version, and outputs match the recorded values.',
      category: 'Technical',
    },
    {
      value: 'q7',
      question: 'Can I use rbee for teaching?',
      answer:
        'Yes! rbee is perfect for teaching. Students can run reproducible assignments using seeds, and you can verify submissions using proof bundles.',
      category: 'Research',
    },
    {
      value: 'q8',
      question: 'Is rbee suitable for grant applications?',
      answer:
        'Absolutely. Include proof bundles in grant applications to demonstrate reproducible methodology and verifiable results.',
      category: 'Research',
    },
  ],
}

// === Final CTA ===

/**
 * Final CTA content - Call to action
 */
export const ctaProps: CTATemplateProps = {
  eyebrow: 'Get Started',
  title: 'Ready to Build Reproducible AI Research?',
  subtitle: 'Join researchers using rbee for verifiable, reproducible AI experiments.',
  primary: {
    label: 'Explore Documentation',
    href: '/docs',
  },
  secondary: {
    label: 'Join Research Community',
    href: '/community',
    variant: 'outline',
  },
  note: 'Open-source • GPL-3.0-or-later • Deterministic seeds • Proof bundles included',
  emphasis: 'gradient',
}
