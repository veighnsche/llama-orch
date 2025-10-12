import type React from 'react'
import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'
import { Suspense } from 'react'
import { Navigation } from '@/components/organisms/Navigation/Navigation'
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
    <html lang="en" suppressHydrationWarning className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="font-sans">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <Navigation />
          <main id="main">
            <Suspense fallback={null}>{children}</Suspense>
          </main>
        </ThemeProvider>
      </body>
    </html>
  )
}
