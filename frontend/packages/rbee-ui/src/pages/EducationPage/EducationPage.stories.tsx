import type { Meta, StoryObj } from '@storybook/react'
import EducationPage from './EducationPage'

const meta = {
  title: 'Pages/EducationPage',
  component: EducationPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Education page targeting students, educators, and career switchers. Showcases hands-on learning with production-grade distributed AI systems, BDD testing, and real-world skills.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EducationPage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Education page with all sections:
 * - Hero with distributed AI visualization
 * - Email capture for educator resources
 * - Problem section (learning gap)
 * - Solution section (real production infrastructure)
 * - Course levels (Beginner, Intermediate, Advanced)
 * - Curriculum modules (6 core modules)
 * - Lab exercises (4-step hands-on labs)
 * - Student types (CS Student, Career Switcher, Researcher)
 * - Student testimonials and outcomes
 * - Learning resources grid
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}
