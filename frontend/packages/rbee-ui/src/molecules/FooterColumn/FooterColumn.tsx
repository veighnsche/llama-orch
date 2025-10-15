import { cn } from '@rbee/ui/utils'
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
  const headingId = `footer-${title.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <div className={className}>
      <h3 id={headingId} className="text-sm font-semibold text-foreground mb-4">
        {title}
      </h3>
      <ul className="space-y-2" aria-labelledby={headingId}>
        {links.map((link, index) => (
          <li key={index}>
            {link.external ? (
              <a
                href={link.href}
                target="_blank"
                rel="noreferrer"
                title="Opens in a new tab"
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                {link.text}
              </a>
            ) : (
              <Link
                href={link.href}
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
                {...(link.href === '#'
                  ? { 'aria-disabled': 'true', onClick: (e: React.MouseEvent) => e.preventDefault() }
                  : {})}
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
