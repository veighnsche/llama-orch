"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Menu, X, Github } from "lucide-react"
import { useState } from "react"

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 text-xl font-semibold text-primary hover:text-primary/80 transition-colors"
          >
            <span className="text-2xl">🐝</span>
            <span>rbee</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <Link href="/features" className="text-muted-foreground hover:text-foreground transition-colors">
              Features
            </Link>
            <Link href="/use-cases" className="text-muted-foreground hover:text-foreground transition-colors">
              Use Cases
            </Link>
            <Link href="/pricing" className="text-muted-foreground hover:text-foreground transition-colors">
              Pricing
            </Link>
            <Link href="/developers" className="text-muted-foreground hover:text-foreground transition-colors">
              For Developers
            </Link>
            <Link href="/gpu-providers" className="text-muted-foreground hover:text-foreground transition-colors">
              For Providers
            </Link>
            <Link href="/enterprise" className="text-muted-foreground hover:text-foreground transition-colors">
              For Enterprise
            </Link>

            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>

            <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">
              Join Waitlist
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden text-muted-foreground hover:text-foreground"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-4 border-t border-border">
            <Link
              href="/features"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Features
            </Link>
            <Link
              href="/use-cases"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Use Cases
            </Link>
            <Link
              href="/pricing"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              Pricing
            </Link>
            <Link
              href="/developers"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Developers
            </Link>
            <Link
              href="/gpu-providers"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Providers
            </Link>
            <Link
              href="/enterprise"
              className="block text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Enterprise
            </Link>
            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-muted-foreground hover:text-foreground transition-colors"
            >
              GitHub
            </a>
            <Button size="sm" className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">
              Join Waitlist
            </Button>
          </div>
        )}
      </div>
    </nav>
  )
}
