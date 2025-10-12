import { cn } from '@/lib/utils'
import Link from 'next/link'

export interface FooterLink {
  href: string
  text: string
  external?: boolean
}

export interface FooterColumnProps {
  title: string
  links: FooterLink[]
  className?: string
}

export function FooterColumn({ title, links, className }: FooterColumnProps) {
  return (
    <div className={className}>
      <h3 className="text-foreground font-bold mb-4">{title}</h3>
      <ul className="space-y-2 text-sm">
        {links.map((link, index) => (
          <li key={index}>
            {link.external ? (
              <a
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.text}
              </a>
            ) : (
              <Link
                href={link.href}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.text}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
