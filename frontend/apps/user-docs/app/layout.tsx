import type { Metadata } from 'next'
// Import order: app CSS (JIT scanning) → UI CSS (tokens) → Nextra theme
import './globals.css'
import '@rbee/ui/styles.css'
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
