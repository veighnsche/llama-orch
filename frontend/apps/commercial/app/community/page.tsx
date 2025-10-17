import { CommunityPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Community - Join the Discussion',
  description:
    'Join the rbee community. Connect with developers, share knowledge, get help, and contribute to the open-source project.',
  keywords: ['rbee community', 'AI community', 'open source community', 'developer community', 'Discord', 'GitHub'],
  alternates: {
    canonical: '/community',
  },
  openGraph: {
    title: 'rbee Community - Connect with Developers',
    description: 'Join the rbee community. Share knowledge, get help, contribute.',
    type: 'website',
    url: 'https://rbee.dev/community',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Community',
    description: 'Join the rbee community. Connect with developers building with AI.',
  },
}

export default function Page() {
  return <CommunityPage />
}
