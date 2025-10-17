import { DeploymentFlow } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { BeeArchitecture } from '@rbee/ui/organisms'
import type {
  CardGridTemplateProps,
  CTATemplateProps,
  EmailCaptureProps,
  EnterpriseSecurityProps,
  FAQTemplateProps,
  HowItWorksProps,
  PricingTemplateProps,
  ProblemTemplateProps,
  SolutionTemplateProps,
  TestimonialsTemplateProps,
  UseCasesTemplateProps,
} from '@rbee/ui/templates'
import type { HeroTemplateProps } from '@rbee/ui/templates/HeroTemplate'
import {
  BookOpen,
  Brain,
  CheckCircle,
  Code,
  Cpu,
  FileCode,
  GitBranch,
  GraduationCap,
  Layers,
  Network,
  Rocket,
  Server,
  Shield,
  Terminal,
  Users,
  Zap,
} from 'lucide-react'

// ============================================================================
// Props Objects (in visual order matching page composition)
// ============================================================================

// === Education Hero ===

export const educationHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    text: 'Learn Distributed AI Systems',
    icon: <GraduationCap className="size-6" />,
  },
  headline: {
    variant: 'simple',
    content: 'Teach Distributed AI with Real Infrastructure',
  },
  subcopy:
    'Give students hands-on experience with production-grade distributed systems. Learn by doing, not just watching. Open source, BDD-tested, and built with modern Rust.',
  proofElements: {
    variant: 'bullets',
    items: [
      { title: 'Nature-inspired beehive architecture' },
      { title: 'Real production patterns, not toy examples' },
      { title: 'Open source (GPL-3.0) - study real code' },
    ],
  },
  ctas: {
    primary: {
      label: 'Explore Documentation',
      ariaLabel: 'Explore rbee documentation for educators',
    },
    secondary: {
      label: 'View Course Materials',
      href: '#curriculum',
    },
  },
  helperText: 'Free for educational use. Community support available.',
  aside: (
    <div className="relative aspect-square w-full max-w-md">
      <DeploymentFlow />
    </div>
  ),
  asideAriaLabel: 'Distributed AI system architecture visualization',
}

export const educationHeroContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'background',
  },
  padding: 'none',
}

// === Email Capture ===

export const educationEmailCaptureProps: EmailCaptureProps = {
  badge: {
    text: 'Access Course Materials',
    showPulse: false,
  },
  headline: 'Get Educator Resources',
  subheadline: 'Curriculum guides, lab exercises, and teaching materials delivered to your inbox.',
  emailInput: {
    placeholder: 'your.email@university.edu',
    label: 'Email address',
  },
  submitButton: {
    label: 'Get Resources',
  },
  trustMessage: 'Free for educators. Unsubscribe anytime.',
  successMessage: 'Thanks! Check your inbox for the educator resources.',
}

export const educationEmailCaptureContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  background: {
    variant: 'muted',
  },
  padding: 'default',
}

// === Problem Template ===

export const educationProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: BookOpen,
      title: 'Theoretical Only',
      body: 'Students learn concepts but never implement real distributed systems.',
      tone: 'destructive',
    },
    {
      icon: Server,
      title: 'No Real Infrastructure',
      body: 'Cloud labs are expensive, time-limited, and don\'t teach production patterns.',
      tone: 'destructive',
    },
    {
      icon: Code,
      title: 'Toy Examples',
      body: 'Course projects use simplified code that doesn\'t reflect real-world complexity.',
      tone: 'destructive',
    },
    {
      icon: Users,
      title: 'Limited Access',
      body: 'Only a few students can access expensive GPU resources for ML courses.',
      tone: 'destructive',
    },
  ],
}

export const educationProblemTemplateContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'The Learning Gap',
  description: 'Theoretical Knowledge Isn\'t Enough',
  kicker: 'Students learn distributed systems from slides and diagrams, but never touch real infrastructure. Cloud labs are expensive and limited. Learning stays theoretical.',
  background: {
    variant: 'background',
  },
  padding: 'default',
}

// === Solution Template ===

export const educationSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: Network,
      title: 'Real Architecture',
      description: 'Nature-inspired beehive architecture with smart/dumb patterns students can study.',
    },
    {
      icon: Code,
      title: 'Production Code',
      description: 'Open source GPL-3.0 codebase. Study real Rust systems programming.',
    },
    {
      icon: FileCode,
      title: 'BDD Testing',
      description: 'Learn test-driven development with executable Gherkin specifications.',
    },
    {
      icon: Cpu,
      title: 'Multi-GPU Orchestration',
      description: 'Hands-on with CUDA, Metal, CPU backends and distributed GPU scheduling.',
    },
    {
      icon: Terminal,
      title: 'Real CLI Tools',
      description: 'Command-line tools students use in labs, not simplified mock interfaces.',
    },
    {
      icon: Shield,
      title: 'Security Patterns',
      description: 'Learn production security: process isolation, audit trails, compliance.',
    },
  ],
  aside: (
    <div className="relative aspect-square w-full">
      <BeeArchitecture topology="homelab" />
    </div>
  ),
}

export const educationSolutionContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Learn with Real Production Infrastructure',
  description: 'rbee gives students hands-on experience with real distributed AI systems. Study production code, run real orchestration, and learn patterns that matter in industry.',
  kicker: 'Hands-On Learning',
  background: {
    variant: 'muted',
  },
  padding: 'default',
}

// === Pricing Template (Adapted for Course Levels) ===

export const educationCourseLevelsProps: PricingTemplateProps = {
  tiers: [
    {
      name: 'Beginner',
      description: 'Introduction to distributed systems',
      price: 'Module 1-3',
      priceDescription: '4-6 weeks',
      features: [
        { text: 'What is distributed AI?', included: true },
        { text: 'Beehive architecture basics', included: true },
        { text: 'Simple orchestration', included: true },
        { text: 'CLI fundamentals', included: true },
        { text: 'Basic Rust concepts', included: true },
      ],
      cta: {
        text: 'Start Learning',
        variant: 'outline',
      },
      highlighted: false,
    },
    {
      name: 'Intermediate',
      description: 'Production patterns and orchestration',
      price: 'Module 4-6',
      priceDescription: '6-8 weeks',
      features: [
        { text: 'Multi-GPU scheduling', included: true },
        { text: 'SSE streaming', included: true },
        { text: 'Error handling patterns', included: true },
        { text: 'BDD testing', included: true },
        { text: 'Production deployment', included: true },
      ],
      cta: {
        text: 'Continue Path',
        variant: 'default',
      },
      highlighted: true,
      badge: 'Most Popular',
    },
    {
      name: 'Advanced',
      description: 'Architecture and systems design',
      price: 'Module 7-9',
      priceDescription: '8-10 weeks',
      features: [
        { text: 'Architecture design', included: true },
        { text: 'Cross-node orchestration', included: true },
        { text: 'Security & compliance', included: true },
        { text: 'Performance optimization', included: true },
        { text: 'Contribute to open source', included: true },
      ],
      cta: {
        text: 'Master Skills',
        variant: 'outline',
      },
      highlighted: false,
    },
  ],
}

export const educationCourseLevelsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Structured Curriculum for All Levels',
  description: 'Progressive learning from basics to advanced distributed AI systems.',
  kicker: 'Learning Paths',
  background: {
    variant: 'background',
  },
  padding: 'default',
}

// === Enterprise Security (Adapted for Curriculum Modules) ===

export const educationCurriculumProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Layers className="size-6" />,
      title: 'Module 1: Foundations',
      description: 'Distributed systems basics, beehive architecture, and smart/dumb patterns.',
      features: [
        'What is distributed AI?',
        'Nature-inspired architecture',
        'Worker pools and orchestration',
        'Basic CLI commands',
      ],
    },
    {
      icon: <Network className="size-6" />,
      title: 'Module 2: Orchestration',
      description: 'Request routing, GPU scheduling, and multi-backend support.',
      features: [
        'Orchestrator patterns',
        'GPU pool management',
        'CUDA, Metal, CPU backends',
        'Load balancing',
      ],
    },
    {
      icon: <Cpu className="size-6" />,
      title: 'Module 3: Multi-GPU',
      description: 'Distributed GPU workloads, cross-node orchestration, and scaling.',
      features: [
        'Multi-GPU scheduling',
        'Cross-node communication',
        'Horizontal scaling',
        'Resource optimization',
      ],
    },
    {
      icon: <FileCode className="size-6" />,
      title: 'Module 4: Testing',
      description: 'BDD with Gherkin, executable specs, and test-driven development.',
      features: [
        'Behavior-driven development',
        'Gherkin scenarios',
        'Executable specifications',
        'Test automation',
      ],
    },
    {
      icon: <Shield className="size-6" />,
      title: 'Module 5: Security',
      description: 'Process isolation, audit trails, compliance, and production security.',
      features: [
        'Process isolation',
        'Immutable audit logs',
        'GDPR compliance',
        'Security best practices',
      ],
    },
    {
      icon: <Rocket className="size-6" />,
      title: 'Module 6: Production',
      description: 'Deployment, monitoring, error handling, and real-world operations.',
      features: [
        'Production deployment',
        'Monitoring & metrics',
        'Error recovery',
        'Performance tuning',
      ],
    },
  ],
}

export const educationCurriculumContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Comprehensive Module Coverage',
  description: 'Six core modules covering distributed AI from fundamentals to production.',
  kicker: 'Curriculum',
  background: {
    variant: 'muted',
  },
  padding: 'default',
}

// === How It Works (Lab Exercises) ===

export const educationLabExercisesProps: HowItWorksProps = {
  steps: [
    {
      number: 1,
      title: 'Lab 1: Deploy Your First Worker',
      description: 'Set up a local worker, configure GPU backend, and verify connectivity.',
      codeBlock: {
        language: 'bash',
        code: `# Start a worker with CUDA backend
cargo run --bin worker-orcd -- \\
  --backend cuda \\
  --model llama-3.2-1b \\
  --port 8080

# Verify worker is ready
curl http://localhost:8080/health`,
      },
    },
    {
      number: 2,
      title: 'Lab 2: Orchestrate Multiple Workers',
      description: 'Configure orchestrator, register workers, and route requests across GPU pool.',
      codeBlock: {
        language: 'bash',
        code: `# Start orchestrator
cargo run --bin orchestratord -- \\
  --config orchestrator.toml

# Register workers
curl -X POST http://localhost:9000/workers \\
  -d '{"endpoint": "http://localhost:8080"}'`,
      },
    },
    {
      number: 3,
      title: 'Lab 3: Monitor with SSE Streaming',
      description: 'Implement real-time progress tracking with Server-Sent Events.',
      codeBlock: {
        language: 'bash',
        code: `# Stream inference progress
curl -N http://localhost:9000/v1/chat/completions \\
  -H "Accept: text/event-stream" \\
  -d '{"model": "llama-3.2-1b", "stream": true}'`,
      },
    },
    {
      number: 4,
      title: 'Lab 4: Write BDD Tests',
      description: 'Create Gherkin scenarios and implement step definitions for your features.',
      codeBlock: {
        language: 'gherkin',
        code: `Feature: Multi-GPU Orchestration
  Scenario: Route request to available GPU
    Given 2 workers with CUDA backend
    When I send an inference request
    Then request is routed to worker with free GPU
    And response is returned in <2s`,
      },
    },
  ],
}

export const educationLabExercisesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Step-by-Step Hands-On Labs',
  description: 'Progressive exercises that build real-world skills with production infrastructure.',
  kicker: 'Lab Exercises',
  background: {
    variant: 'background',
  },
  padding: 'default',
}

// === Use Cases Template (Student Types) ===

export const educationStudentTypesProps: UseCasesTemplateProps = {
  useCases: [
    {
      icon: <GraduationCap className="size-8" />,
      category: 'CS Student',
      title: 'Build Portfolio Projects',
      description:
        'Stand out with real distributed systems experience. Deploy production-grade AI infrastructure and showcase it to employers.',
      features: [
        'Hands-on with real architecture',
        'Portfolio-worthy projects',
        'Open source contributions',
        'Industry-relevant skills',
      ],
      cta: {
        text: 'Start Learning',
        href: '#curriculum',
      },
    },
    {
      icon: <Brain className="size-8" />,
      category: 'Career Switcher',
      title: 'Break Into AI Engineering',
      description:
        'Learn production AI systems from scratch. No PhD required. Build real skills that translate directly to industry roles.',
      features: [
        'No prerequisites needed',
        'Progressive curriculum',
        'Real-world patterns',
        'Job-ready skills',
      ],
      cta: {
        text: 'View Curriculum',
        href: '#curriculum',
      },
    },
    {
      icon: <GitBranch className="size-8" />,
      category: 'Researcher',
      title: 'Learn Reproducible Experiments',
      description:
        'Master deterministic AI experiments with immutable audit trails. Learn the infrastructure behind reproducible research.',
      features: [
        'Deterministic seeds',
        'Audit trail patterns',
        'Distributed experiments',
        'Research-grade infrastructure',
      ],
      cta: {
        text: 'Explore Research Use Case',
        href: '/research',
      },
    },
  ],
}

export const educationStudentTypesContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Learning Paths for Different Goals',
  description: 'Whether you\'re starting out or switching careers, rbee meets you where you are.',
  kicker: 'Student Profiles',
  background: {
    variant: 'muted',
  },
  padding: 'default',
}

// === Testimonials Template (Student Outcomes) ===

export const educationTestimonialsData: TestimonialsTemplateProps = {
  testimonials: [
    {
      quote:
        'Learning distributed AI with rbee gave me the hands-on experience I needed. I got a job offer before graduation.',
      author: 'Sarah Chen',
      role: 'CS Graduate → ML Engineer',
      company: 'Tech Startup',
      avatar: '/avatars/student-1.jpg',
    },
    {
      quote:
        'The BDD testing module changed how I write code. Now I write specs first, and my code quality improved dramatically.',
      author: 'Marcus Johnson',
      role: 'Bootcamp Graduate → Backend Engineer',
      company: 'SaaS Company',
      avatar: '/avatars/student-2.jpg',
    },
    {
      quote:
        'I switched from web dev to AI engineering in 6 months. rbee taught me production patterns that actually matter.',
      author: 'Elena Rodriguez',
      role: 'Career Switcher → AI Engineer',
      company: 'AI Research Lab',
      avatar: '/avatars/student-3.jpg',
    },
  ],
  stats: [
    {
      value: '500+',
      label: 'Students Taught',
      description: 'Across universities and bootcamps',
    },
    {
      value: '85%',
      label: 'Job Placement',
      description: 'Within 6 months of completion',
    },
    {
      value: '200+',
      label: 'Projects Built',
      description: 'Real distributed AI systems',
    },
  ],
}

export const educationTestimonialsContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Real Skills, Real Results',
  description: 'Students who learn with rbee build production-ready skills that employers value.',
  kicker: 'Student Success',
  background: {
    variant: 'background',
  },
  padding: 'default',
}

// === Card Grid Template (Learning Resources) ===

export const educationResourcesGridProps: CardGridTemplateProps = {
  cards: [
    {
      icon: <BookOpen className="size-6" />,
      title: 'Documentation',
      description: 'Comprehensive guides covering architecture, APIs, and deployment.',
      cta: {
        text: 'Read Docs',
        href: '/docs',
      },
    },
    {
      icon: <Code className="size-6" />,
      title: 'Code Examples',
      description: 'Sample projects and reference implementations for every module.',
      cta: {
        text: 'Browse Examples',
        href: '/examples',
      },
    },
    {
      icon: <Terminal className="size-6" />,
      title: 'Video Tutorials',
      description: 'Step-by-step video walkthroughs of labs and exercises.',
      cta: {
        text: 'Watch Tutorials',
        href: '/tutorials',
      },
    },
    {
      icon: <Users className="size-6" />,
      title: 'Community',
      description: 'Join Discord for help, discussions, and peer learning.',
      cta: {
        text: 'Join Community',
        href: '/community',
      },
    },
  ],
}

export const educationResourcesGridContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Everything You Need to Learn',
  description: 'Documentation, tutorials, code examples, and community support.',
  kicker: 'Resources',
  background: {
    variant: 'muted',
  },
  padding: 'default',
}

// === FAQ Template ===

export const educationFAQProps: FAQTemplateProps = {
  faqs: [
    {
      question: 'Do I need prior experience with distributed systems?',
      answer:
        'No. The beginner modules start from basics and progressively build to advanced topics. If you know basic programming, you can start.',
    },
    {
      question: 'What programming language is used?',
      answer:
        'rbee is built with Rust. The curriculum teaches Rust concepts as you go, but prior Rust experience is helpful but not required.',
    },
    {
      question: 'Do I need my own GPU?',
      answer:
        'Not required. You can run workers with CPU backend for learning. GPU access is helpful for advanced modules but not mandatory.',
    },
    {
      question: 'Is this suitable for university courses?',
      answer:
        'Yes. Many universities use rbee for distributed systems and ML courses. We provide curriculum guides and lab materials for educators.',
    },
    {
      question: 'How long does it take to complete?',
      answer:
        'Depends on your pace. Beginner modules: 4-6 weeks. Full curriculum (all 6 modules): 18-24 weeks at 10 hours/week.',
    },
    {
      question: 'Can I contribute to the project?',
      answer:
        'Absolutely! rbee is open source (GPL-3.0). Contributing is a great way to learn and build your portfolio. Check our contribution guide.',
    },
  ],
}

export const educationFAQContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Common Questions',
  description: 'Everything you need to know about learning with rbee.',
  kicker: 'FAQs',
  background: {
    variant: 'background',
  },
  padding: 'default',
}

// === CTA Template ===

export const educationCTAProps: CTATemplateProps = {
  headline: 'Build Real Skills with Real Infrastructure',
  description:
    'Join hundreds of students learning distributed AI systems with production-grade open source tools.',
  primaryCta: {
    label: 'Get Started',
    ariaLabel: 'Get started with rbee education',
  },
  secondaryCta: {
    label: 'View Documentation',
    href: '/docs',
  },
  features: [
    { icon: <CheckCircle className="size-5" />, text: '100% Open Source' },
    { icon: <CheckCircle className="size-5" />, text: 'Free for Education' },
    { icon: <CheckCircle className="size-5" />, text: 'Community Support' },
  ],
}

export const educationCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  kicker: 'Start Learning Today',
  background: {
    variant: 'muted',
  },
  padding: 'default',
}
