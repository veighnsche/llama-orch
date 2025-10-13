'use client'

import { MessageCircle } from 'lucide-react'
import { GitHubIcon } from '@/components/atoms/GitHubIcon/GitHubIcon'
import { BrandLogo } from '@/components/molecules/BrandLogo/BrandLogo'
import { FooterColumn } from '@/components/molecules'
import { Button } from '@/components/atoms/Button/Button'
import { Input } from '@/components/atoms/Input/Input'

export function Footer() {
  const handleNewsletterSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    // TODO: Implement newsletter subscription
    const formData = new FormData(e.currentTarget)
    const email = formData.get('email')
    console.log('Newsletter signup:', email)
  }

  return (
    <footer className="relative bg-background text-muted-foreground pt-14 pb-10 before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-border/60">
      <div className="container mx-auto px-4 space-y-12">
        {/* 1. Utility Bar */}
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between animate-fade-in">
          <div className="flex-1 max-w-2xl">
            <h3 className="text-sm font-semibold text-foreground mb-1">Stay in the loop</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Releases, roadmap, and self-hosting tips. 1–2 emails/month.
            </p>
            <form onSubmit={handleNewsletterSubmit} className="flex gap-2 max-w-md">
              <Input
                type="email"
                name="email"
                required
                placeholder="you@company.com"
                aria-label="Email address"
                className="h-10"
              />
              <Button type="submit" className="h-10 shrink-0">
                Subscribe
              </Button>
            </form>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline" size="sm" className="h-9">
              <a href="https://github.com/veighnsche/llama-orch/tree/main/docs">View Docs</a>
            </Button>
            <Button asChild variant="outline" size="sm" className="h-9">
              <a href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noreferrer" title="Opens in a new tab">
                Star on GitHub
              </a>
            </Button>
            <Button asChild size="sm" className="h-9">
              <a href="https://discord.gg/rbee" target="_blank" rel="noreferrer" title="Opens in a new tab">
                Join Discord
              </a>
            </Button>
          </div>
        </div>

        {/* 2. Sitemap Grid */}
        <nav aria-label="Footer" className="grid gap-8 md:grid-cols-4 animate-fade-in">
          <FooterColumn
            title="Product"
            links={[
              {
                href: 'https://github.com/veighnsche/llama-orch/tree/main/docs',
                text: 'Documentation',
                external: true,
              },
              {
                href: 'https://github.com/veighnsche/llama-orch/tree/main/QUICKSTART_INFERENCE.md',
                text: 'Quickstart (15 min)',
                external: true,
              },
              { href: 'https://github.com/veighnsche/llama-orch', text: 'GitHub', external: true },
              { href: 'https://github.com/veighnsche/llama-orch/milestones', text: 'Roadmap', external: true },
              { href: 'https://github.com/veighnsche/llama-orch/releases', text: 'Changelog', external: true },
            ]}
          />

          <FooterColumn
            title="Community"
            links={[
              { href: 'https://discord.gg/rbee', text: 'Discord', external: true },
              {
                href: 'https://github.com/veighnsche/llama-orch/discussions',
                text: 'GitHub Discussions',
                external: true,
              },
              { href: 'https://x.com/rbee', text: 'X (Twitter)', external: true },
              { href: '/blog', text: 'Blog' },
            ]}
          />

          <FooterColumn
            title="Company"
            links={[
              { href: '/about', text: 'About' },
              { href: '/#pricing', text: 'Pricing' },
              { href: '/contact', text: 'Contact' },
              { href: 'https://github.com/veighnsche/llama-orch/issues', text: 'Support', external: true },
            ]}
          />

          <FooterColumn
            title="Legal"
            links={[
              { href: '/legal/privacy', text: 'Privacy Policy' },
              { href: '/legal/terms', text: 'Terms of Service' },
              {
                href: 'https://github.com/veighnsche/llama-orch/blob/main/LICENSE',
                text: 'License (GPL-3.0)',
                external: true,
              },
              { href: 'https://github.com/veighnsche/llama-orch/security', text: 'Security', external: true },
            ]}
          />
        </nav>

        {/* 3. Bottom Bar */}
        <div className="border-t border-border pt-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between animate-fade-in-up">
          <div className="flex items-center gap-3">
            <BrandLogo size="sm" />
            <p className="text-sm text-muted-foreground">
              © 2025 rbee. Built with 🍯 by developers, for developers.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <GitHubIcon className="size-5" />
            </a>
            <a
              href="https://github.com/veighnsche/llama-orch/discussions"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub Discussions"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <MessageCircle className="size-5" />
            </a>
            <a
              href="https://x.com/rbee"
              target="_blank"
              rel="noreferrer"
              aria-label="X (Twitter)"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <svg className="size-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
              </svg>
            </a>
            <a
              href="https://discord.gg/rbee"
              target="_blank"
              rel="noreferrer"
              aria-label="Discord"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <svg className="size-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
