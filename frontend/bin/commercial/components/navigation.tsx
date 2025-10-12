"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Menu, X, Github } from "lucide-react";
import { useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { NavLink } from "@/components/primitives";

export function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

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
            <NavLink href="/features">Features</NavLink>
            <NavLink href="/use-cases">Use Cases</NavLink>
            <NavLink href="/pricing">Pricing</NavLink>
            <NavLink href="/developers">For Developers</NavLink>
            <NavLink href="/gpu-providers">For Providers</NavLink>
            <NavLink href="/enterprise">For Enterprise</NavLink>
            <NavLink href="https://github.com/veighnsche/llama-orch/tree/main/docs">Docs</NavLink>

            <div className="flex items-center gap-2">
              <a
                href="https://github.com/veighnsche/llama-orch"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center size-9 rounded-md text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5" />
              </a>

              <ThemeToggle />
            </div>

            <Button
              size="sm"
              className="bg-primary hover:bg-primary/80 text-primary-foreground"
            >
              Join Waitlist
            </Button>
          </div>

          {/* Mobile Menu Button & Theme Toggle */}
          <div className="md:hidden flex items-center gap-2">
            <ThemeToggle />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-muted-foreground hover:text-foreground"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-4 border-t border-border">
            <NavLink
              href="/features"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              Features
            </NavLink>
            <NavLink
              href="/use-cases"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              Use Cases
            </NavLink>
            <NavLink
              href="/pricing"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              Pricing
            </NavLink>
            <NavLink
              href="/developers"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Developers
            </NavLink>
            <NavLink
              href="/gpu-providers"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Providers
            </NavLink>
            <NavLink
              href="/enterprise"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              For Enterprise
            </NavLink>
            <NavLink
              href="https://github.com/veighnsche/llama-orch/tree/main/docs"
              variant="mobile"
              onClick={() => setMobileMenuOpen(false)}
            >
              Docs
            </NavLink>
            <a
              href="https://github.com/veighnsche/llama-orch"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-muted-foreground hover:text-foreground transition-colors"
            >
              GitHub
            </a>
            <Button
              size="sm"
              className="w-full bg-primary hover:bg-primary/80 text-primary-foreground"
            >
              Join Waitlist
            </Button>
          </div>
        )}
      </div>
    </nav>
  );
}
