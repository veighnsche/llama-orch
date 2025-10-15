import { test, expect } from '@playwright/experimental-ct-react';
import { Button } from './Button';

test.describe('Button', () => {
  test('renders with default props', async ({ mount }) => {
    const component = await mount(<Button>Click me</Button>);
    await expect(component).toBeVisible();
    await expect(component).toContainText('Click me');
  });

  test('renders with default variant (primary)', async ({ mount }) => {
    const component = await mount(<Button>Default</Button>);
    await expect(component).toBeVisible();
    await expect(component).toContainText('Default');
  });

  test('renders with secondary variant', async ({ mount }) => {
    const component = await mount(<Button variant="secondary">Secondary</Button>);
    await expect(component).toBeVisible();
  });

  test('renders with outline variant', async ({ mount }) => {
    const component = await mount(<Button variant="outline">Outline</Button>);
    await expect(component).toBeVisible();
  });

  test('renders with ghost variant', async ({ mount }) => {
    const component = await mount(<Button variant="ghost">Ghost</Button>);
    await expect(component).toBeVisible();
  });

  test('renders with link variant', async ({ mount }) => {
    const component = await mount(<Button variant="link">Link</Button>);
    await expect(component).toBeVisible();
  });

  test('renders with destructive variant', async ({ mount }) => {
    const component = await mount(<Button variant="destructive">Delete</Button>);
    await expect(component).toBeVisible();
  });

  test('renders with different sizes', async ({ mount }) => {
    const small = await mount(<Button size="sm">Small</Button>);
    await expect(small).toBeVisible();

    const medium = await mount(<Button size="default">Default</Button>);
    await expect(medium).toBeVisible();

    const large = await mount(<Button size="lg">Large</Button>);
    await expect(large).toBeVisible();

    const icon = await mount(<Button size="icon">🔍</Button>);
    await expect(icon).toBeVisible();
  });

  test('handles click events', async ({ mount }) => {
    let clicked = false;
    const component = await mount(
      <Button onClick={() => { clicked = true; }}>Click me</Button>
    );
    await component.click();
    expect(clicked).toBe(true);
  });

  test('renders as disabled', async ({ mount }) => {
    const component = await mount(<Button disabled>Disabled</Button>);
    await expect(component).toBeDisabled();
  });

  test('renders with asChild prop', async ({ mount }) => {
    const component = await mount(
      <Button asChild>
        <a href="/test">Link Button</a>
      </Button>
    );
    await expect(component.locator('a')).toBeVisible();
    await expect(component.locator('a')).toHaveAttribute('href', '/test');
  });
});
