import type { Metadata } from 'next'
import '@rbee/ui/styles.css'
import './globals.css'
import 'nextra-theme-docs/style.css'

export const metadata: Metadata = {
	title: 'rbee Documentation',
	description: 'Documentation for rbee - Private LLM Hosting in the Netherlands',
}

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode
}>) {
	return (
		<html lang="en" suppressHydrationWarning>
			<body>{children}</body>
		</html>
	)
}
