import type { Meta, StoryObj } from '@storybook/react'
import { TemplateBackground } from './TemplateBackground'

const meta = {
  title: 'Organisms/TemplateBackground',
  component: TemplateBackground,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TemplateBackground>

export default meta
type Story = StoryObj<typeof meta>

const SampleContent = () => (
  <div className="container mx-auto px-4 py-24">
    <div className="max-w-3xl mx-auto text-center space-y-6">
      <h2 className="text-4xl font-bold">Template Background Showcase</h2>
      <p className="text-xl text-muted-foreground">
        This demonstrates the various background variants available for templates.
      </p>
      <div className="grid grid-cols-3 gap-4 pt-8">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="h-32 bg-card rounded-lg border flex items-center justify-center">
            <span className="text-muted-foreground">Card {i}</span>
          </div>
        ))}
      </div>
    </div>
  </div>
)

// Solid Colors
export const Background: Story = {
  args: {
    variant: 'background',
    children: <SampleContent />,
  },
}

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: <SampleContent />,
  },
}

export const Card: Story = {
  args: {
    variant: 'card',
    children: <SampleContent />,
  },
}

export const Muted: Story = {
  args: {
    variant: 'muted',
    children: <SampleContent />,
  },
}

// Gradients
export const GradientPrimary: Story = {
  args: {
    variant: 'gradient-primary',
    children: <SampleContent />,
  },
}

export const GradientSecondary: Story = {
  args: {
    variant: 'gradient-secondary',
    children: <SampleContent />,
  },
}

export const GradientRadial: Story = {
  args: {
    variant: 'gradient-radial',
    children: <SampleContent />,
  },
}

export const GradientMesh: Story = {
  args: {
    variant: 'gradient-mesh',
    children: <SampleContent />,
  },
}

export const GradientWarm: Story = {
  args: {
    variant: 'gradient-warm',
    children: <SampleContent />,
  },
}

export const GradientCool: Story = {
  args: {
    variant: 'gradient-cool',
    children: <SampleContent />,
  },
}

// Patterns - Small
export const PatternDotsSmall: Story = {
  args: {
    variant: 'pattern-dots',
    patternSize: 'small',
    patternOpacity: 15,
    children: <SampleContent />,
  },
}

export const PatternGridSmall: Story = {
  args: {
    variant: 'pattern-grid',
    patternSize: 'small',
    patternOpacity: 12,
    children: <SampleContent />,
  },
}

export const PatternHoneycombSmall: Story = {
  args: {
    variant: 'pattern-honeycomb',
    patternSize: 'small',
    patternOpacity: 10,
    children: <SampleContent />,
  },
}

export const PatternWavesSmall: Story = {
  args: {
    variant: 'pattern-waves',
    patternSize: 'small',
    patternOpacity: 12,
    children: <SampleContent />,
  },
}

export const PatternCircuitSmall: Story = {
  args: {
    variant: 'pattern-circuit',
    patternSize: 'small',
    patternOpacity: 15,
    children: <SampleContent />,
  },
}

export const PatternDiagonalSmall: Story = {
  args: {
    variant: 'pattern-diagonal',
    patternSize: 'small',
    patternOpacity: 8,
    children: <SampleContent />,
  },
}

// Patterns - Medium (default)
export const PatternDotsMedium: Story = {
  args: {
    variant: 'pattern-dots',
    patternSize: 'medium',
    patternOpacity: 12,
    children: <SampleContent />,
  },
}

export const PatternGridMedium: Story = {
  args: {
    variant: 'pattern-grid',
    patternSize: 'medium',
    patternOpacity: 10,
    children: <SampleContent />,
  },
}

export const PatternHoneycombMedium: Story = {
  args: {
    variant: 'pattern-honeycomb',
    patternSize: 'medium',
    patternOpacity: 8,
    children: <SampleContent />,
  },
}

export const PatternWavesMedium: Story = {
  args: {
    variant: 'pattern-waves',
    patternSize: 'medium',
    patternOpacity: 10,
    children: <SampleContent />,
  },
}

export const PatternCircuitMedium: Story = {
  args: {
    variant: 'pattern-circuit',
    patternSize: 'medium',
    patternOpacity: 12,
    children: <SampleContent />,
  },
}

export const PatternDiagonalMedium: Story = {
  args: {
    variant: 'pattern-diagonal',
    patternSize: 'medium',
    patternOpacity: 6,
    children: <SampleContent />,
  },
}

// Patterns - Large
export const PatternDotsLarge: Story = {
  args: {
    variant: 'pattern-dots',
    patternSize: 'large',
    patternOpacity: 10,
    children: <SampleContent />,
  },
}

export const PatternGridLarge: Story = {
  args: {
    variant: 'pattern-grid',
    patternSize: 'large',
    patternOpacity: 8,
    children: <SampleContent />,
  },
}

export const PatternHoneycombLarge: Story = {
  args: {
    variant: 'pattern-honeycomb',
    patternSize: 'large',
    patternOpacity: 6,
    children: <SampleContent />,
  },
}

export const PatternWavesLarge: Story = {
  args: {
    variant: 'pattern-waves',
    patternSize: 'large',
    patternOpacity: 8,
    children: <SampleContent />,
  },
}

export const PatternCircuitLarge: Story = {
  args: {
    variant: 'pattern-circuit',
    patternSize: 'large',
    patternOpacity: 10,
    children: <SampleContent />,
  },
}

export const PatternDiagonalLarge: Story = {
  args: {
    variant: 'pattern-diagonal',
    patternSize: 'large',
    patternOpacity: 5,
    children: <SampleContent />,
  },
}

// Combined Effects
export const GradientWithOverlay: Story = {
  args: {
    variant: 'gradient-primary',
    overlayOpacity: 20,
    overlayColor: 'black',
    children: <SampleContent />,
  },
}

export const PatternWithBlur: Story = {
  args: {
    variant: 'pattern-honeycomb',
    patternSize: 'large',
    patternOpacity: 15,
    blur: true,
    children: <SampleContent />,
  },
}

export const CustomDecoration: Story = {
  args: {
    variant: 'background',
    decoration: (
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="custom-stars" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
              <circle cx="50" cy="50" r="2" fill="currentColor" />
              <circle cx="25" cy="25" r="1.5" fill="currentColor" />
              <circle cx="75" cy="75" r="1.5" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#custom-stars)" className="text-primary" />
        </svg>
      </div>
    ),
    children: <SampleContent />,
  },
}
