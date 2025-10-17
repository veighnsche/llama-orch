import { Button } from '@rbee/ui/atoms/Button'
import { Card, CardContent, CardFooter } from '@rbee/ui/atoms/Card'
import { DeploymentFlow } from '@rbee/ui/atoms'
import type { TemplateContainerProps } from '@rbee/ui/molecules'
import { IconCardHeader } from '@rbee/ui/molecules'
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
} from 'lucide-react'
import Link from 'next/link'

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
  paddingY: 'lg',
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
  paddingY: 'xl',
}

// === Problem Template ===

export const educationProblemTemplateProps: ProblemTemplateProps = {
  items: [
    {
      icon: <BookOpen className="h-6 w-6" />,
      title: 'Theoretical Only',
      body: 'Students learn concepts but never implement real distributed systems.',
      tone: 'destructive',
    },
    {
      icon: <Server className="h-6 w-6" />,
      title: 'No Real Infrastructure',
      body: 'Cloud labs are expensive, time-limited, and don\'t teach production patterns.',
      tone: 'destructive',
    },
    {
      icon: <Code className="h-6 w-6" />,
      title: 'Toy Examples',
      body: 'Course projects use simplified code that doesn\'t reflect real-world complexity.',
      tone: 'destructive',
    },
    {
      icon: <Users className="h-6 w-6" />,
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
  paddingY: 'xl',
}

// === Solution Template ===

export const educationSolutionProps: SolutionTemplateProps = {
  features: [
    {
      icon: <Network className="h-8 w-8" />,
      title: 'Real Architecture',
      body: 'Nature-inspired beehive architecture with smart/dumb patterns students can study.',
    },
    {
      icon: <Code className="h-8 w-8" />,
      title: 'Production Code',
      body: 'Open source GPL-3.0 codebase. Study real Rust systems programming.',
    },
    {
      icon: <FileCode className="h-8 w-8" />,
      title: 'BDD Testing',
      body: 'Learn test-driven development with executable Gherkin specifications.',
    },
    {
      icon: <Cpu className="h-8 w-8" />,
      title: 'Multi-GPU Orchestration',
      body: 'Hands-on with CUDA, Metal, CPU backends and distributed GPU scheduling.',
    },
    {
      icon: <Terminal className="h-8 w-8" />,
      title: 'Real CLI Tools',
      body: 'Command-line tools students use in labs, not simplified mock interfaces.',
    },
    {
      icon: <Shield className="h-8 w-8" />,
      title: 'Security Patterns',
      body: 'Learn production security: process isolation, audit trails, compliance.',
    },
  ],
  aside: (
    <div className="relative aspect-square w-full">
      <BeeArchitecture 
        topology={{
          mode: 'single-pc',
          hostLabel: 'Student Laptop',
          workers: [
            { id: 'w1', label: 'GPU 1', kind: 'cuda' },
            { id: 'w2', label: 'GPU 2', kind: 'cuda' },
          ]
        }}
      />
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
  paddingY: 'xl',
}

// === Pricing Template (Adapted for Course Levels) ===

export const educationCourseLevelsProps: PricingTemplateProps = {
  tiers: [
    {
      title: 'Beginner',
      price: 'Module 1-3',
      period: '4-6 weeks',
      features: [
        'What is distributed AI?',
        'Beehive architecture basics',
        'Simple orchestration',
        'CLI fundamentals',
        'Basic Rust concepts',
      ],
      ctaText: 'Start Learning',
      ctaHref: '#curriculum',
      ctaVariant: 'outline',
      highlighted: false,
    },
    {
      title: 'Intermediate',
      price: 'Module 4-6',
      period: '6-8 weeks',
      features: [
        'Multi-GPU scheduling',
        'SSE streaming',
        'Error handling patterns',
        'BDD testing',
        'Production deployment',
      ],
      ctaText: 'Continue Path',
      ctaHref: '#curriculum',
      ctaVariant: 'default',
      highlighted: true,
      badge: 'Most Popular',
    },
    {
      title: 'Advanced',
      price: 'Module 7-9',
      period: '8-10 weeks',
      features: [
        'Architecture design',
        'Cross-node orchestration',
        'Security & compliance',
        'Performance optimization',
        'Contribute to open source',
      ],
      ctaText: 'Master Skills',
      ctaHref: '#curriculum',
      ctaVariant: 'outline',
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
  paddingY: 'xl',
}

// === Enterprise Security (Adapted for Curriculum Modules) ===

export const educationCurriculumProps: EnterpriseSecurityProps = {
  securityCards: [
    {
      icon: <Layers className="size-6" />,
      title: 'Module 1: Foundations',
      subtitle: 'Distributed Systems Basics',
      intro: 'Distributed systems basics, beehive architecture, and smart/dumb patterns.',
      bullets: [
        'What is distributed AI?',
        'Nature-inspired architecture',
        'Worker pools and orchestration',
        'Basic CLI commands',
      ],
      docsHref: '/docs/foundations',
    },
    {
      icon: <Network className="size-6" />,
      title: 'Module 2: Orchestration',
      subtitle: 'Request Routing & Scheduling',
      intro: 'Request routing, GPU scheduling, and multi-backend support.',
      bullets: [
        'Orchestrator patterns',
        'GPU pool management',
        'CUDA, Metal, CPU backends',
        'Load balancing',
      ],
      docsHref: '/docs/orchestration',
    },
    {
      icon: <Cpu className="size-6" />,
      title: 'Module 3: Multi-GPU',
      subtitle: 'Distributed Workloads',
      intro: 'Distributed GPU workloads, cross-node orchestration, and scaling.',
      bullets: [
        'Multi-GPU scheduling',
        'Cross-node communication',
        'Horizontal scaling',
        'Resource optimization',
      ],
      docsHref: '/docs/multi-gpu',
    },
    {
      icon: <FileCode className="size-6" />,
      title: 'Module 4: Testing',
      subtitle: 'Behavior-Driven Development',
      intro: 'BDD with Gherkin, executable specs, and test-driven development.',
      bullets: [
        'Behavior-driven development',
        'Gherkin scenarios',
        'Executable specifications',
        'Test automation',
      ],
      docsHref: '/docs/testing',
    },
    {
      icon: <Shield className="size-6" />,
      title: 'Module 5: Security',
      subtitle: 'Production Security Patterns',
      intro: 'Process isolation, audit trails, compliance, and production security.',
      bullets: [
        'Process isolation',
        'Immutable audit logs',
        'GDPR compliance',
        'Security best practices',
      ],
      docsHref: '/docs/security',
    },
    {
      icon: <Rocket className="size-6" />,
      title: 'Module 6: Production',
      subtitle: 'Real-World Operations',
      intro: 'Deployment, monitoring, error handling, and real-world operations.',
      bullets: [
        'Production deployment',
        'Monitoring & metrics',
        'Error recovery',
        'Performance tuning',
      ],
      docsHref: '/docs/production',
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
  paddingY: 'xl',
}

// === How It Works (Lab Exercises) ===

export const educationLabExercisesProps: HowItWorksProps = {
  steps: [
    {
      number: 1,
      label: 'Lab 1: Deploy Your First Worker',
      block: {
        kind: 'code',
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
      label: 'Lab 2: Orchestrate Multiple Workers',
      block: {
        kind: 'code',
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
      label: 'Lab 3: Monitor with SSE Streaming',
      block: {
        kind: 'code',
        language: 'bash',
        code: `# Stream inference progress
curl -N http://localhost:9000/v1/chat/completions \\
  -H "Accept: text/event-stream" \\
  -d '{"model": "llama-3.2-1b", "stream": true}'`,
      },
    },
    {
      number: 4,
      label: 'Lab 4: Write BDD Tests',
      block: {
        kind: 'code',
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
  paddingY: 'xl',
}

// === Use Cases Template (Student Types) ===

export const educationStudentTypesProps: UseCasesTemplateProps = {
  items: [
    {
      icon: <GraduationCap className="size-8" />,
      title: 'CS Student: Build Portfolio Projects',
      scenario: 'One gaming PC with a single GPU. Want to run AI locally for personal use.',
      solution: 'Stand out with real distributed systems experience. Deploy production-grade AI infrastructure and showcase it to employers.',
      outcome: 'Hands-on with real architecture. Portfolio-worthy projects. Open source contributions. Industry-relevant skills.',
      tags: ['Portfolio', 'Real Experience', 'Open Source'],
      cta: {
        label: 'Start Learning',
        href: '#curriculum',
      },
    },
    {
      icon: <Brain className="size-8" />,
      title: 'Career Switcher: Break Into AI Engineering',
      scenario: 'Switching careers into AI engineering. Need practical skills, not just theory.',
      solution: 'Learn production AI systems from scratch. No PhD required. Build real skills that translate directly to industry roles.',
      outcome: 'No prerequisites needed. Progressive curriculum. Real-world patterns. Job-ready skills.',
      tags: ['Career Change', 'Practical Skills', 'Job Ready'],
      cta: {
        label: 'View Curriculum',
        href: '#curriculum',
      },
    },
    {
      icon: <GitBranch className="size-8" />,
      title: 'Researcher: Learn Reproducible Experiments',
      scenario: 'Need to run reproducible AI experiments with audit trails for research.',
      solution: 'Master deterministic AI experiments with immutable audit trails. Learn the infrastructure behind reproducible research.',
      outcome: 'Deterministic seeds. Audit trail patterns. Distributed experiments. Research-grade infrastructure.',
      tags: ['Research', 'Reproducibility', 'Audit Trails'],
      cta: {
        label: 'Explore Research Use Case',
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
  paddingY: 'xl',
}

// === Testimonials Template (Student Outcomes) ===

export const educationTestimonialsData: TestimonialsTemplateProps = {
  testimonials: [
    {
      quote:
        'Learning distributed AI with rbee gave me the hands-on experience I needed. I got a job offer before graduation.',
      author: 'Sarah Chen',
      role: 'CS Graduate → ML Engineer at Tech Startup',
      avatar: '/avatars/student-1.jpg',
    },
    {
      quote:
        'The BDD testing module changed how I write code. Now I write specs first, and my code quality improved dramatically.',
      author: 'Marcus Johnson',
      role: 'Bootcamp Graduate → Backend Engineer at SaaS Company',
      avatar: '/avatars/student-2.jpg',
    },
    {
      quote:
        'I switched from web dev to AI engineering in 6 months. rbee taught me production patterns that actually matter.',
      author: 'Elena Rodriguez',
      role: 'Career Switcher → AI Engineer at AI Research Lab',
      avatar: '/avatars/student-3.jpg',
    },
  ],
  stats: [
    {
      value: '500+',
      label: 'Students Taught Across Universities',
    },
    {
      value: '85%',
      label: 'Job Placement Within 6 Months',
    },
    {
      value: '200+',
      label: 'Real Distributed AI Projects Built',
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
  paddingY: 'xl',
}

// === Card Grid Template (Learning Resources) ===

export const educationResourcesGridProps: CardGridTemplateProps = {
  cards: [
    <Card key="docs">
      <IconCardHeader icon={<BookOpen className="size-6" />} title="Documentation" />
      <CardContent className="px-6 py-0">
        <p className="text-sm text-muted-foreground">Comprehensive guides covering architecture, APIs, and deployment.</p>
      </CardContent>
      <CardFooter className="mt-4">
        <Button asChild variant="ghost" size="sm">
          <Link href="/docs">Read Docs</Link>
        </Button>
      </CardFooter>
    </Card>,
    <Card key="examples">
      <IconCardHeader icon={<Code className="size-6" />} title="Code Examples" />
      <CardContent className="px-6 py-0">
        <p className="text-sm text-muted-foreground">Sample projects and reference implementations for every module.</p>
      </CardContent>
      <CardFooter className="mt-4">
        <Button asChild variant="ghost" size="sm">
          <Link href="/examples">Browse Examples</Link>
        </Button>
      </CardFooter>
    </Card>,
    <Card key="tutorials">
      <IconCardHeader icon={<Terminal className="size-6" />} title="Video Tutorials" />
      <CardContent className="px-6 py-0">
        <p className="text-sm text-muted-foreground">Step-by-step video walkthroughs of labs and exercises.</p>
      </CardContent>
      <CardFooter className="mt-4">
        <Button asChild variant="ghost" size="sm">
          <Link href="/tutorials">Watch Tutorials</Link>
        </Button>
      </CardFooter>
    </Card>,
    <Card key="community">
      <IconCardHeader icon={<Users className="size-6" />} title="Community" />
      <CardContent className="px-6 py-0">
        <p className="text-sm text-muted-foreground">Join Discord for help, discussions, and peer learning.</p>
      </CardContent>
      <CardFooter className="mt-4">
        <Button asChild variant="ghost" size="sm">
          <Link href="/community">Join Community</Link>
        </Button>
      </CardFooter>
    </Card>,
  ],
}

export const educationResourcesGridContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'Everything You Need to Learn',
  description: 'Documentation, tutorials, code examples, and community support.',
  kicker: 'Resources',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
}

// === FAQ Template ===

export const educationFAQProps: FAQTemplateProps = {
  badgeText: 'Education FAQ',
  categories: ['Getting Started', 'Technical', 'Curriculum'],
  faqItems: [
    {
      value: 'prior-experience',
      question: 'Do I need prior experience with distributed systems?',
      answer:
        'No. The beginner modules start from basics and progressively build to advanced topics. If you know basic programming, you can start.',
      category: 'Getting Started',
    },
    {
      value: 'programming-language',
      question: 'What programming language is used?',
      answer:
        'rbee is built with Rust. The curriculum teaches Rust concepts as you go, but prior Rust experience is helpful but not required.',
      category: 'Technical',
    },
    {
      value: 'gpu-required',
      question: 'Do I need my own GPU?',
      answer:
        'Not required. You can run workers with CPU backend for learning. GPU access is helpful for advanced modules but not mandatory.',
      category: 'Technical',
    },
    {
      value: 'university-courses',
      question: 'Is this suitable for university courses?',
      answer:
        'Yes. Many universities use rbee for distributed systems and ML courses. We provide curriculum guides and lab materials for educators.',
      category: 'Curriculum',
    },
    {
      value: 'completion-time',
      question: 'How long does it take to complete?',
      answer:
        'Depends on your pace. Beginner modules: 4-6 weeks. Full curriculum (all 6 modules): 18-24 weeks at 10 hours/week.',
      category: 'Curriculum',
    },
    {
      value: 'contribute',
      question: 'Can I contribute to the project?',
      answer:
        'Absolutely! rbee is open source (GPL-3.0). Contributing is a great way to learn and build your portfolio. Check our contribution guide.',
      category: 'Getting Started',
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
  paddingY: 'xl',
}

// === CTA Template ===

export const educationCTAProps: CTATemplateProps = {
  eyebrow: 'Start Learning Today',
  title: 'Build Real Skills with Real Infrastructure',
  subtitle:
    'Join hundreds of students learning distributed AI systems with production-grade open source tools.',
  primary: {
    label: 'Get Started',
    href: '#curriculum',
  },
  secondary: {
    label: 'View Documentation',
    href: '/docs',
  },
  note: '100% Open Source • Free for Education • Community Support',
  align: 'center',
  emphasis: 'gradient',
}

export const educationCTAContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: null,
  kicker: 'Start Learning Today',
  background: {
    variant: 'muted',
  },
  paddingY: 'xl',
}
