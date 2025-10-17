// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { BulletListItem } from '@rbee/ui/molecules'
import { CheckItem } from './CheckItem'

const meta: Meta<typeof CheckItem> = {
  title: 'Atoms/CheckItem (Deprecated)',
  component: CheckItem,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## ⚠️ DEPRECATED

**CheckItem is deprecated.** Use \`BulletListItem\` with \`variant="check"\` and \`showPlate={false}\` instead.

### Migration Guide

\`\`\`tsx
// Before:
<CheckItem>Feature text</CheckItem>

// After:
<BulletListItem variant="check" showPlate={false} title="Feature text" />
\`\`\`

This component is now a shim that wraps BulletListItem for backwards compatibility.
        `,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof CheckItem>

export const DeprecatedUsage: Story = {
  args: {
    children: 'JWT with 15-minute expiry',
  },
}

export const MigrationExample: Story = {
  render: () => (
    <div className="space-y-6 max-w-md">
      <div>
        <h3 className="text-sm font-semibold mb-2">❌ Old (Deprecated):</h3>
        <ul className="space-y-2">
          <CheckItem>JWT with 15-minute expiry</CheckItem>
          <CheckItem>Refresh token rotation</CheckItem>
          <CheckItem>Device fingerprinting</CheckItem>
        </ul>
      </div>
      <div>
        <h3 className="text-sm font-semibold mb-2">✅ New (Recommended):</h3>
        <ul className="space-y-2">
          <BulletListItem variant="check" showPlate={false} title="JWT with 15-minute expiry" />
          <BulletListItem variant="check" showPlate={false} title="Refresh token rotation" />
          <BulletListItem variant="check" showPlate={false} title="Device fingerprinting" />
        </ul>
      </div>
    </div>
  ),
}
