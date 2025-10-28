import type { Meta, StoryObj } from '@storybook/react'
import {
  Download,
  Mail,
  Play,
  Save,
  Send,
  Share2,
  Trash2,
  Upload,
} from 'lucide-react'
import { DropdownMenuItem, DropdownMenuSeparator } from '../DropdownMenu'
import { SplitButton } from './SplitButton'

const meta = {
  title: 'Atoms/SplitButton',
  component: SplitButton,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'secondary', 'ghost'],
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg'],
    },
    disabled: {
      control: 'boolean',
    },
  },
} satisfies Meta<typeof SplitButton>

export default meta
type Story = StoryObj<typeof meta>

// Default story with primary action
export const Default: Story = {
  args: {
    children: 'Send Email',
    icon: <Send className="h-4 w-4" />,
    onClick: () => alert('Primary action: Send Email'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Schedule send')}>
          <Mail className="mr-2 h-4 w-4" />
          Schedule Send
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save as draft')}>
          <Save className="mr-2 h-4 w-4" />
          Save as Draft
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert('Discard')} variant="destructive">
          <Trash2 className="mr-2 h-4 w-4" />
          Discard
        </DropdownMenuItem>
      </>
    ),
  },
}

// Service control (like Queen/Hive start buttons)
export const ServiceControl: Story = {
  args: {
    children: 'Start',
    icon: <Play className="h-4 w-4" />,
    onClick: () => alert('Starting service...'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Stop service')}>
          <Trash2 className="mr-2 h-4 w-4 text-danger" />
          Stop
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert('Install')}>
          <Download className="mr-2 h-4 w-4" />
          Install
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Update')}>
          <Upload className="mr-2 h-4 w-4" />
          Update
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert('Uninstall')} variant="destructive">
          <Trash2 className="mr-2 h-4 w-4" />
          Uninstall
        </DropdownMenuItem>
      </>
    ),
  },
}

// Destructive variant
export const Destructive: Story = {
  args: {
    children: 'Delete',
    variant: 'destructive',
    icon: <Trash2 className="h-4 w-4" />,
    onClick: () => alert('Delete immediately'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Move to trash')}>
          <Trash2 className="mr-2 h-4 w-4" />
          Move to Trash
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Delete permanently')}>
          <Trash2 className="mr-2 h-4 w-4" />
          Delete Permanently
        </DropdownMenuItem>
      </>
    ),
  },
}

// Outline variant
export const Outline: Story = {
  args: {
    children: 'Share',
    variant: 'outline',
    icon: <Share2 className="h-4 w-4" />,
    onClick: () => alert('Share via link'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Share via email')}>
          <Mail className="mr-2 h-4 w-4" />
          Share via Email
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Share on social')}>
          <Share2 className="mr-2 h-4 w-4" />
          Share on Social
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Copy link')}>
          <Download className="mr-2 h-4 w-4" />
          Copy Link
        </DropdownMenuItem>
      </>
    ),
  },
}

// Secondary variant
export const Secondary: Story = {
  args: {
    children: 'Save',
    variant: 'secondary',
    icon: <Save className="h-4 w-4" />,
    onClick: () => alert('Save'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Save as...')}>
          <Save className="mr-2 h-4 w-4" />
          Save As...
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save all')}>
          <Save className="mr-2 h-4 w-4" />
          Save All
        </DropdownMenuItem>
      </>
    ),
  },
}

// Ghost variant
export const Ghost: Story = {
  args: {
    children: 'Download',
    variant: 'ghost',
    icon: <Download className="h-4 w-4" />,
    onClick: () => alert('Download'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Download as PDF')}>
          <Download className="mr-2 h-4 w-4" />
          Download as PDF
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Download as CSV')}>
          <Download className="mr-2 h-4 w-4" />
          Download as CSV
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Download as JSON')}>
          <Download className="mr-2 h-4 w-4" />
          Download as JSON
        </DropdownMenuItem>
      </>
    ),
  },
}

// Small size
export const Small: Story = {
  args: {
    children: 'Send',
    size: 'sm',
    icon: <Send className="h-4 w-4" />,
    onClick: () => alert('Send'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Schedule')}>
          <Mail className="mr-2 h-4 w-4" />
          Schedule
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save draft')}>
          <Save className="mr-2 h-4 w-4" />
          Save Draft
        </DropdownMenuItem>
      </>
    ),
  },
}

// Large size
export const Large: Story = {
  args: {
    children: 'Send Email',
    size: 'lg',
    icon: <Send className="h-4 w-4" />,
    onClick: () => alert('Send'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Schedule')}>
          <Mail className="mr-2 h-4 w-4" />
          Schedule Send
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save draft')}>
          <Save className="mr-2 h-4 w-4" />
          Save as Draft
        </DropdownMenuItem>
      </>
    ),
  },
}

// Without icon
export const WithoutIcon: Story = {
  args: {
    children: 'Send',
    onClick: () => alert('Send'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Schedule')}>
          <Mail className="mr-2 h-4 w-4" />
          Schedule Send
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save draft')}>
          <Save className="mr-2 h-4 w-4" />
          Save as Draft
        </DropdownMenuItem>
      </>
    ),
  },
}

// Disabled state
export const Disabled: Story = {
  args: {
    children: 'Send',
    icon: <Send className="h-4 w-4" />,
    disabled: true,
    onClick: () => alert('This should not fire'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Schedule')}>
          <Mail className="mr-2 h-4 w-4" />
          Schedule Send
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Save draft')}>
          <Save className="mr-2 h-4 w-4" />
          Save as Draft
        </DropdownMenuItem>
      </>
    ),
  },
}

// Full width (like in cards)
export const FullWidth: Story = {
  args: {
    children: 'Start Service',
    icon: <Play className="h-4 w-4" />,
    onClick: () => alert('Starting...'),
    className: 'w-full',
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Stop')}>
          <Trash2 className="mr-2 h-4 w-4" />
          Stop
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Restart')}>
          <Upload className="mr-2 h-4 w-4" />
          Restart
        </DropdownMenuItem>
      </>
    ),
  },
  decorators: [
    (Story) => (
      <div style={{ width: '400px' }}>
        <Story />
      </div>
    ),
  ],
}

// Multiple actions showcase
export const ComplexMenu: Story = {
  args: {
    children: 'Deploy',
    icon: <Upload className="h-4 w-4" />,
    onClick: () => alert('Deploy to production'),
    dropdownContent: (
      <>
        <DropdownMenuItem onClick={() => alert('Deploy to staging')}>
          <Upload className="mr-2 h-4 w-4" />
          Deploy to Staging
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('Deploy to development')}>
          <Upload className="mr-2 h-4 w-4" />
          Deploy to Development
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert('Preview changes')}>
          <Download className="mr-2 h-4 w-4" />
          Preview Changes
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => alert('View logs')}>
          <Mail className="mr-2 h-4 w-4" />
          View Logs
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => alert('Rollback')} variant="destructive">
          <Trash2 className="mr-2 h-4 w-4" />
          Rollback
        </DropdownMenuItem>
      </>
    ),
  },
}
