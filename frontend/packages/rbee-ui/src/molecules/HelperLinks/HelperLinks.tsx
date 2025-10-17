import { ChevronRight } from 'lucide-react'
import Link from 'next/link'

export interface HelperLink {
  label: string
  href: string
}

export interface HelperLinksProps {
  links: HelperLink[]
}

export function HelperLinks({ links }: HelperLinksProps) {
  if (!links || links.length === 0) return null

  return (
    <div className="mx-auto text-center">
      {links.map((link, index) => (
        <span key={index}>
          {index > 0 && <span className="mx-3 text-muted-foreground/50">·</span>}
          <Link
            href={link.href}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
          >
            {link.label}
            <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </span>
      ))}
    </div>
  )
}
