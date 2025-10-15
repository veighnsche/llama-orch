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
  title: 'rbee - Build with AI. Own Your Infrastructure.',
  description:
    'Open-source AI orchestration platform. Orchestrate inference across your home network hardware with zero ongoing costs. OpenAI-compatible API.',
  generator: 'v0.app',
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
