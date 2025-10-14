import { DollarSign, Shield, BarChart3, Clock, Sliders, Zap } from 'lucide-react'
import { FeatureTabsSection } from '@rbee/ui/organisms'

export function ProvidersFeatures() {
  return (
    <FeatureTabsSection
      title="Everything You Need to Maximize Earnings"
      subtitle="Professional-grade tools to manage your GPU fleet and optimize your passive income."
      items={[
        {
          id: 'pricing',
          title: 'Flexible Pricing Control',
          description: 'Set your own hourly rates based on GPU model, demand, and your preferences.',
          icon: <DollarSign className="h-4 w-4" />,
          benefit: { text: 'Dynamic pricing based on demand with automatic adjustments', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Pricing Configuration',
            content: `{
  "gpu": "RTX 4090",
  "base_rate": 1.50,
  "min_rate": 1.00,
  "max_rate": 3.00,
  "demand_multiplier": true,
  "schedule": {
    "weekday": 1.5,
    "weekend": 2.0
  }
}`,
          },
        },
        {
          id: 'availability',
          title: 'Availability Management',
          description: 'Control exactly when your GPUs are available for rent.',
          icon: <Clock className="h-4 w-4" />,
          benefit: { text: 'Set availability windows and priority modes', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Availability Schedule',
            content: `{
  "weekday": "09:00-17:00",
  "weekend": "all-day",
  "vacation_mode": false,
  "priority_mode": "my_usage_first",
  "auto_pause_gaming": true
}`,
          },
        },
        {
          id: 'security',
          title: 'Security & Privacy',
          description: 'Your data and hardware are protected with enterprise-grade security.',
          icon: <Shield className="h-4 w-4" />,
          benefit: { text: 'Sandboxed execution with encrypted communication', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Security Features',
            content: `✓ Sandboxed execution (no file access)
✓ Encrypted communication (TLS 1.3)
✓ No access to personal data
✓ Malware scanning on all jobs
✓ Automatic security updates
✓ Insurance coverage included`,
          },
        },
        {
          id: 'analytics',
          title: 'Earnings Dashboard',
          description: 'Track your earnings, utilization, and performance in real‑time.',
          icon: <BarChart3 className="h-4 w-4" />,
          benefit: { text: 'Real‑time earnings tracking with historical charts', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Earnings Summary',
            content: `Today:        €42.50
This Week:    €287.30
This Month:   €1,124.80

Utilization:  78%
Avg Rate:     €1.85/hr
Top GPU:      RTX 4090 (€524/mo)`,
          },
        },
        {
          id: 'limits',
          title: 'Usage Limits',
          description: 'Set limits to protect your hardware and control costs.',
          icon: <Sliders className="h-4 w-4" />,
          benefit: { text: 'Temperature monitoring and automatic cooldown periods', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Hardware Protection',
            content: `{
  "max_hours_per_day": 18,
  "temp_limit": 80,
  "power_cap": 350,
  "cooldown_minutes": 15,
  "warranty_mode": true
}`,
          },
        },
        {
          id: 'performance',
          title: 'Performance Optimization',
          description: 'Maximize your earnings with automatic optimization.',
          icon: <Zap className="h-4 w-4" />,
          benefit: { text: 'Automatic model selection and load balancing', tone: 'primary' },
          example: {
            kind: 'code',
            title: 'Optimization Stats',
            content: `Idle Detection:     ✓ Active
Auto-Start:         ✓ Enabled
Load Balancing:     2 GPUs
Priority Queue:     High-paying jobs first
Benchmark Score:    9,847 (top 5%)
Earnings Boost:     +23% vs. baseline`,
          },
        },
      ]}
    />
  )
}
