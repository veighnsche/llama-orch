import { MessageCircle } from 'lucide-react'
import { GitHubIcon } from '@/components/atoms/GitHubIcon/GitHubIcon'
import { BrandLogo } from '@/components/molecules/BrandLogo/BrandLogo'
import { FooterColumn } from '@/components/molecules'

export function Footer() {
  return (
    <footer className="bg-background text-muted-foreground py-16">
      <div className="container mx-auto px-4">
        <div className="mb-12">
          <BrandLogo size="lg" className="mb-8" />
        </div>
        
        <div className="grid md:grid-cols-4 gap-8 mb-12">
          <FooterColumn
            title="Product"
            links={[
              {
                href: 'https://github.com/veighnsche/llama-orch/tree/main/docs',
                text: 'Documentation',
                external: true,
              },
              { href: 'https://github.com/veighnsche/llama-orch', text: 'GitHub Repository', external: true },
              { href: 'https://github.com/veighnsche/llama-orch/milestones', text: 'Roadmap', external: true },
              { href: 'https://github.com/veighnsche/llama-orch/releases', text: 'Changelog', external: true },
            ]}
          />

          <FooterColumn
            title="Community"
            links={[
              { href: '#', text: 'Discord' },
              {
                href: 'https://github.com/veighnsche/llama-orch/discussions',
                text: 'GitHub Discussions',
                external: true,
              },
              { href: '#', text: 'Twitter/X' },
              { href: '#', text: 'Blog' },
            ]}
          />

          <FooterColumn
            title="Company"
            links={[
              { href: '#', text: 'About' },
              { href: '/#pricing', text: 'Pricing' },
              { href: '#', text: 'Contact Sales' },
              { href: 'https://github.com/veighnsche/llama-orch/issues', text: 'Support', external: true },
            ]}
          />

          <FooterColumn
            title="Legal"
            links={[
              { href: '#', text: 'Privacy Policy' },
              { href: '#', text: 'Terms of Service' },
              {
                href: 'https://github.com/veighnsche/llama-orch/blob/main/LICENSE',
                text: 'License (GPL)',
                external: true,
              },
              { href: 'https://github.com/veighnsche/llama-orch/security', text: 'Security', external: true },
            ]}
          />
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-border pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm">© 2025 rbee. Built with 🍯 by developers, for developers.</p>
          <div className="flex gap-4">
            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors"
            >
              <GitHubIcon className="h-5 w-5" />
            </a>
            <a href="#" className="hover:text-primary transition-colors">
              <MessageCircle className="h-5 w-5" />
            </a>
            <a href="#" className="hover:text-primary transition-colors">
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
