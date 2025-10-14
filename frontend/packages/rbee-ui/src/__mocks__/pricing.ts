// Mock pricing data for Storybook stories

export const mockPricingTiers = [
  {
    id: 'starter',
    name: 'Starter',
    price: '€49',
    period: '/month',
    description: 'Perfect for small teams getting started with private LLM hosting',
    features: [
      'Up to 2 GPU workers',
      '100GB model storage',
      'Basic monitoring',
      'Email support',
      'Community access',
    ],
    cta: 'Start Free Trial',
    highlighted: false,
  },
  {
    id: 'professional',
    name: 'Professional',
    price: '€199',
    period: '/month',
    description: 'For growing teams that need more power and flexibility',
    features: [
      'Up to 10 GPU workers',
      '500GB model storage',
      'Advanced monitoring',
      'Priority support',
      'Custom integrations',
      'SLA guarantee',
    ],
    cta: 'Start Free Trial',
    highlighted: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For organizations with advanced requirements',
    features: [
      'Unlimited GPU workers',
      'Unlimited storage',
      'Dedicated support',
      'Custom deployment',
      'On-premise option',
      'Training & onboarding',
    ],
    cta: 'Contact Sales',
    highlighted: false,
  },
];
