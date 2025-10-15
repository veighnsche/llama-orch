import type { Metadata } from 'next'
import type React from 'react'
// 🚨 TURBOREPO PATTERN: Import pre-built UI CSS FIRST, then app CSS
// ✅ This is the idiomatic way - UI package builds its own CSS
// ❌ NEVER use @source to scan UI package files from here
// ✅ All fonts are loaded in @rbee/ui/styles.css (Geist Sans, Geist Mono, Source Serif 4)
import '@rbee/ui/styles.css'
import './globals.css'
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
			<body className="font-sans">
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
