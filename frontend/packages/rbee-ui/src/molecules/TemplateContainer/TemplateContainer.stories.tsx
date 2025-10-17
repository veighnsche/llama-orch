import type { Meta, StoryObj } from "@storybook/react";
import { Shield, Sparkles } from "lucide-react";
import { Button } from "@rbee/ui/atoms";
import { TemplateContainer } from "./TemplateContainer";

const meta = {
  title: "Molecules/TemplateContainer",
  component: TemplateContainer,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof TemplateContainer>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Comprehensive story showcasing ALL available props of TemplateContainer.
 * This demonstrates the full capabilities of the component including:
 * - Header elements (eyebrow, kicker, title, description, actions)
 * - Layout options (align, layout, bleed, paddingY, maxWidth)
 * - Background variants and decorations
 * - Multiple CTA patterns (bottom CTAs, CTA banner, CTA rail, footer CTA)
 * - Security guarantees section
 * - Disclaimer and ribbon
 * - Semantic heading levels and dividers
 */
export const AllPropsShowcase: Story = {
  args: {
    // Header elements
    eyebrow: "Enterprise Solutions",
    kicker: "Introducing our most powerful platform yet",
    kickerVariant: "default",
    title: "Complete Template Container Showcase",
    description:
      "This story demonstrates every single prop available in the TemplateContainer component, from header elements to footer CTAs, security guarantees, and everything in between.",
    
    // Actions in header
    actions: (
      <div className="flex gap-2">
        <Button variant="outline" size="sm">
          Learn More
        </Button>
        <Button size="sm">Get Started</Button>
      </div>
    ),

    // Layout & styling
    align: "center",
    layout: "stack",
    background: {
      variant: "gradient-mesh",
      decoration: (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl" />
        </div>
      ),
      overlayOpacity: 10,
      overlayColor: "black",
    },
    bleed: false,
    paddingY: "2xl",
    maxWidth: "6xl",
    divider: true,

    // Semantic HTML
    headlineLevel: 2,
    headingId: "showcase-section",

    // Main content
    children: (
      <div className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-lg border bg-card">
            <h3 className="font-semibold mb-2">Feature One</h3>
            <p className="text-sm text-muted-foreground">
              Comprehensive layout options with flexible alignment and spacing controls.
            </p>
          </div>
          <div className="p-6 rounded-lg border bg-card">
            <h3 className="font-semibold mb-2">Feature Two</h3>
            <p className="text-sm text-muted-foreground">
              Multiple CTA patterns including banners, rails, and footer sections.
            </p>
          </div>
          <div className="p-6 rounded-lg border bg-card">
            <h3 className="font-semibold mb-2">Feature Three</h3>
            <p className="text-sm text-muted-foreground">
              Built-in security guarantees and disclaimer components for trust signals.
            </p>
          </div>
        </div>
      </div>
    ),

    // Security Guarantees
    securityGuarantees: {
      heading: "Enterprise-Grade Security",
      stats: [
        {
          value: "99.99%",
          label: "Uptime SLA",
          ariaLabel: "99.99 percent uptime service level agreement",
        },
        {
          value: "ISO 27001",
          label: "Certified",
          ariaLabel: "ISO 27001 certified security",
        },
        {
          value: "24/7",
          label: "Monitoring",
          ariaLabel: "24/7 security monitoring",
        },
      ],
      footnote: "All data encrypted at rest and in transit with AES-256 encryption",
    },

    // Footer CTA
    footerCTA: {
      message: "Ready to get started with our platform?",
      ctas: [
        {
          label: "Start Free Trial",
          href: "#trial",
          variant: "default",
        },
        {
          label: "Schedule Demo",
          href: "#demo",
          variant: "outline",
        },
      ],
    },

    // CTA Rail
    ctaRail: {
      heading: "Take the next step",
      description: "Join thousands of teams already using our platform",
      buttons: [
        {
          text: "Get Started",
          href: "#start",
          variant: "default",
          ariaLabel: "Get started with our platform",
        },
        {
          text: "View Pricing",
          href: "#pricing",
          variant: "outline",
          ariaLabel: "View pricing plans",
        },
      ],
      links: [
        {
          text: "Documentation",
          href: "#docs",
        },
        {
          text: "API Reference",
          href: "#api",
        },
        {
          text: "Support",
          href: "#support",
        },
      ],
      footnote: "No credit card required for trial",
    },

    // CTA Banner
    ctaBanner: {
      copy: "Special offer: Get 20% off your first year with annual billing",
      primary: {
        label: "Claim Offer",
        href: "#offer",
        ariaLabel: "Claim 20% discount offer",
      },
      secondary: {
        label: "Learn More",
        href: "#details",
        ariaLabel: "Learn more about the offer",
      },
    },

    // Ribbon
    ribbon: {
      text: "Protected by €1M insurance coverage",
    },

    // Bottom CTAs
    ctas: {
      label: "Choose your plan",
      primary: {
        label: "Start Enterprise Trial",
        href: "#enterprise",
        ariaLabel: "Start enterprise trial",
      },
      secondary: {
        label: "Compare Plans",
        href: "#compare",
        ariaLabel: "Compare pricing plans",
      },
      caption: "All plans include 14-day money-back guarantee",
    },

    // Disclaimer
    disclaimer: {
      text: "This is a demonstration of all available props. Actual usage may vary based on your specific needs. Please consult documentation for best practices.",
      variant: "info",
      showIcon: true,
    },
  },
};

/**
 * Minimal configuration showing required props only
 */
export const Minimal: Story = {
  args: {
    title: "Minimal Template Container",
    children: (
      <div className="p-8 text-center">
        <p className="text-muted-foreground">
          This shows the minimum required props: title and children.
        </p>
      </div>
    ),
  },
};

/**
 * Split layout with actions in a two-column header
 */
export const SplitLayout: Story = {
  args: {
    title: "Split Layout Example",
    description: "Actions appear in the right column on medium+ screens",
    layout: "split",
    align: "start",
    actions: (
      <div className="flex gap-2">
        <Button variant="outline">Cancel</Button>
        <Button>Save Changes</Button>
      </div>
    ),
    children: (
      <div className="p-8 border rounded-lg">
        <p className="text-muted-foreground">Main content area</p>
      </div>
    ),
  },
};

/**
 * Destructive variant with warning kicker
 */
export const DestructiveVariant: Story = {
  args: {
    title: "Critical System Alert",
    kicker: "Immediate action required",
    kickerVariant: "destructive",
    description: "Your attention is needed to resolve this issue",
    bgVariant: "destructive-gradient",
    children: (
      <div className="p-6 border border-destructive/30 rounded-lg bg-destructive/5">
        <p className="text-sm">
          This demonstrates the destructive gradient background and destructive kicker variant.
        </p>
      </div>
    ),
    ctas: {
      primary: {
        label: "Resolve Now",
        href: "#resolve",
      },
      secondary: {
        label: "View Details",
        href: "#details",
      },
    },
  },
};

/**
 * Full-width bleed layout
 */
export const BleedLayout: Story = {
  args: {
    title: "Full-Width Background",
    description: "The background extends edge-to-edge while content stays constrained",
    bgVariant: "muted",
    bleed: true,
    children: (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-lg border bg-card">
          <h3 className="font-semibold mb-2">Contained Content</h3>
          <p className="text-sm text-muted-foreground">
            Content respects max-width while background bleeds
          </p>
        </div>
        <div className="p-6 rounded-lg border bg-card">
          <h3 className="font-semibold mb-2">Flexible Layout</h3>
          <p className="text-sm text-muted-foreground">
            Perfect for hero sections and full-width features
          </p>
        </div>
      </div>
    ),
  },
};
