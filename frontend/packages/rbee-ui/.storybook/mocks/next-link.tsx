import type React from 'react'

// Mock Next.js Link component for Storybook
export default function Link({
  href,
  children,
  ...props
}: {
  href: string
  children: React.ReactNode
  [key: string]: any
}) {
  return (
    <a href={href} {...props}>
      {children}
    </a>
  )
}
