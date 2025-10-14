import type { Meta, StoryObj } from '@storybook/react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from './Card';
import { Button } from '../atoms/Button';
import { Badge } from '../atoms/Badge';

const meta = {
  title: 'Molecules/Card',
  component: Card,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <Card style={{ width: '400px' }}>
      <CardHeader>
        <CardTitle>Card Title</CardTitle>
        <CardDescription>This is a card description</CardDescription>
      </CardHeader>
      <CardContent>
        <p>Card content goes here. You can put any content inside.</p>
      </CardContent>
    </Card>
  ),
};

export const WithFooter: Story = {
  render: () => (
    <Card style={{ width: '400px' }}>
      <CardHeader>
        <CardTitle>Card with Footer</CardTitle>
        <CardDescription>This card has action buttons in the footer</CardDescription>
      </CardHeader>
      <CardContent>
        <p>This is the main content of the card. It can contain text, images, or other components.</p>
      </CardContent>
      <CardFooter style={{ gap: '0.5rem' }}>
        <Button size="sm">Confirm</Button>
        <Button size="sm" variant="outline">Cancel</Button>
      </CardFooter>
    </Card>
  ),
};

export const WithBadge: Story = {
  render: () => (
    <Card style={{ width: '400px' }}>
      <CardHeader>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CardTitle>Feature Card</CardTitle>
          <Badge>New</Badge>
        </div>
        <CardDescription>A new feature announcement</CardDescription>
      </CardHeader>
      <CardContent>
        <p>This card announces a new feature with a badge in the header.</p>
      </CardContent>
    </Card>
  ),
};

export const FeatureGrid: Story = {
  render: () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem', maxWidth: '800px' }}>
      <Card>
        <CardHeader>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CardTitle>Private Infrastructure</CardTitle>
            <Badge>Secure</Badge>
          </div>
          <CardDescription>Your data never leaves your infrastructure</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Deploy models on your own hardware or use our managed service.</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <CardTitle>GDPR Compliant</CardTitle>
            <Badge variant="secondary">EU</Badge>
          </div>
          <CardDescription>Built for European data regulations</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Full compliance with Dutch and European privacy laws.</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>High Performance</CardTitle>
          <CardDescription>Optimized for GPU acceleration</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Fast inference with state-of-the-art optimization techniques.</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Easy to Deploy</CardTitle>
          <CardDescription>Simple setup and configuration</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Get started in minutes with our straightforward deployment process.</p>
        </CardContent>
      </Card>
    </div>
  ),
};

export const MinimalCard: Story = {
  render: () => (
    <Card style={{ width: '300px' }}>
      <CardContent style={{ paddingTop: '1.5rem' }}>
        <p>A minimal card with just content, no header or footer.</p>
      </CardContent>
    </Card>
  ),
};

export const LongContent: Story = {
  render: () => (
    <Card style={{ width: '400px' }}>
      <CardHeader>
        <CardTitle>Long Content Example</CardTitle>
        <CardDescription>This card contains longer text content</CardDescription>
      </CardHeader>
      <CardContent>
        <p style={{ marginBottom: '1rem' }}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
          incididunt ut labore et dolore magna aliqua.
        </p>
        <p style={{ marginBottom: '1rem' }}>
          Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut 
          aliquip ex ea commodo consequat.
        </p>
        <p>
          Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
          eu fugiat nulla pariatur.
        </p>
      </CardContent>
      <CardFooter>
        <Button variant="outline">Read More</Button>
      </CardFooter>
    </Card>
  ),
};
