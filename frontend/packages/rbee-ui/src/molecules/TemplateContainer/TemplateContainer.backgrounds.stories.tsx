import type { Meta, StoryObj } from '@storybook/react'
import { TemplateContainer } from './TemplateContainer'

const meta = {
  title: 'Molecules/TemplateContainer/Backgrounds',
  component: TemplateContainer,
  parameters: {
    layout: 'fullscreen',
  },
} satisfies Meta<typeof TemplateContainer>

export default meta
type Story = StoryObj<typeof meta>

const SampleContent = () => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
    <div className="p-6 rounded-lg border bg-card">
      <h3 className="font-semibold mb-2">Feature One</h3>
      <p className="text-sm text-muted-foreground">Sample feature description</p>
    </div>
    <div className="p-6 rounded-lg border bg-card">
      <h3 className="font-semibold mb-2">Feature Two</h3>
      <p className="text-sm text-muted-foreground">Sample feature description</p>
    </div>
    <div className="p-6 rounded-lg border bg-card">
      <h3 className="font-semibold mb-2">Feature Three</h3>
      <p className="text-sm text-muted-foreground">Sample feature description</p>
    </div>
  </div>
)

export const GradientPrimary: Story = {
  args: {
    title: 'Gradient Primary Background',
    description: 'Using gradient-primary variant',
    background: {
      variant: 'gradient-primary',
    },
    children: <SampleContent />,
  },
}

export const GradientMesh: Story = {
  args: {
    title: 'Gradient Mesh Background',
    description: 'Using gradient-mesh variant',
    background: {
      variant: 'gradient-mesh',
    },
    children: <SampleContent />,
  },
}

export const GradientRadial: Story = {
  args: {
    title: 'Gradient Radial Background',
    description: 'Using gradient-radial variant',
    background: {
      variant: 'gradient-radial',
    },
    children: <SampleContent />,
  },
}

export const WithDecoration: Story = {
  args: {
    title: 'Background with Decoration',
    description: 'Custom SVG pattern decoration',
    background: {
      variant: 'background',
      decoration: (
        <svg className="absolute inset-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      ),
    },
    children: <SampleContent />,
  },
}

export const WithOverlay: Story = {
  args: {
    title: 'Background with Overlay',
    description: 'Primary background with 20% black overlay',
    background: {
      variant: 'primary',
      overlayOpacity: 20,
      overlayColor: 'black',
    },
    children: <SampleContent />,
  },
}

export const WithBlur: Story = {
  args: {
    title: 'Background with Blur',
    description: 'Blurred decoration elements',
    background: {
      variant: 'gradient-mesh',
      decoration: (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-96 h-96 bg-primary/30 rounded-full" />
        </div>
      ),
      blur: true,
    },
    children: <SampleContent />,
  },
}

export const ComplexBackground: Story = {
  args: {
    title: 'Complex Background Example',
    description: 'Gradient + decorations + overlay + blur',
    background: {
      variant: 'gradient-radial',
      decoration: (
        <>
          <div className="absolute top-0 left-0 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-secondary/20 rounded-full blur-3xl" />
        </>
      ),
      overlayOpacity: 10,
      overlayColor: 'black',
    },
    children: <SampleContent />,
  },
}

export const SubtleBorder: Story = {
  args: {
    title: 'Subtle Border Background',
    description: 'Background with top border line',
    background: {
      variant: 'subtle-border',
    },
    children: <SampleContent />,
  },
}

export const PrimaryBrand: Story = {
  args: {
    title: 'Primary Brand Background',
    description: 'Full primary color background',
    background: {
      variant: 'primary',
    },
    children: <SampleContent />,
  },
}

export const LegacyBgVariant: Story = {
  args: {
    title: 'Legacy bgVariant (Deprecated)',
    description: 'Using old bgVariant prop - still works for backward compatibility',
    background: {
      variant: 'gradient-destructive',
    },
    children: <SampleContent />,
  },
}
