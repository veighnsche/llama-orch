import { DevOpsPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for DevOps - Production-Ready AI Orchestration',
  description:
    'Production-ready AI orchestration. Cascading shutdown, health monitoring, multi-node SSH control. Lifecycle management and observability.',
  keywords: [
    'AI DevOps',
    'production AI infrastructure',
    'AI orchestration platform',
    'AI monitoring',
    'distributed AI deployment',
    'lifecycle management',
  ],
  alternates: {
    canonical: '/industries/devops',
  },
  openGraph: {
    title: 'rbee for DevOps - Production AI Orchestration',
    description: 'Cascading shutdown, health monitoring, lifecycle management. Production-ready.',
    type: 'website',
    url: 'https://rbee.dev/industries/devops',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for DevOps - Production AI',
    description: 'Production-ready AI orchestration with lifecycle management.',
  },
}

export default function Page() {
  return <DevOpsPage />
}
