import type { Metadata } from 'next'
import type React from 'react'
// 🚨 TURBOREPO PATTERN: Import app CSS (with JIT scanning), then UI CSS (pre-built tokens)
// ✅ App CSS: Enables arbitrary values like translate-y-[2rem] in app components
// ✅ UI CSS: Provides design tokens and component styles
// ✅ All fonts are loaded in @rbee/ui/styles.css (Geist Sans, Geist Mono, Source Serif 4)
import './globals.css'
import '@rbee/ui/styles.css'
import { Footer, Navigation } from '@rbee/ui/organisms'
import { Suspense } from 'react'
import { ThemeProvider } from '@/components/providers/ThemeProvider/ThemeProvider'

export const metadata: Metadata = {
  title: 'rbee - OpenAI-Compatible AI Infrastructure | Self-Hosted LLMs',
  description:
    'Run LLMs on YOUR hardware with rbee. OpenAI-compatible API, zero ongoing costs, complete privacy. CUDA, Metal, CPU support. Build AI on your terms.',
  keywords: [
    'self-hosted AI',
    'OpenAI alternative',
    'GPU orchestration',
    'private LLM',
    'GDPR-compliant AI',
    'multi-GPU inference',
    'local AI',
    'rbee',
  ],
  authors: [{ name: 'rbee' }],
  creator: 'rbee',
  publisher: 'rbee',
  metadataBase: new URL('https://rbee.dev'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'rbee - Own Your AI Infrastructure',
    description: 'OpenAI-compatible AI platform running on your hardware. Zero API fees, complete privacy.',
    type: 'website',
    url: 'https://rbee.dev',
    siteName: 'rbee',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee - OpenAI-Compatible AI Infrastructure',
    description: 'Run LLMs on YOUR hardware. Zero fees, complete privacy.',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-serif">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <Navigation />
          <main id="main">
            <Suspense fallback={null}>{children}</Suspense>
          </main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  )
}
