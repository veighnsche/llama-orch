/**
 * ⚠️ CRITICAL: DO NOT run `pnpm test:ct` with blocking=true!
 * See PLAYWRIGHT_USAGE.md for correct usage patterns.
 */

import { test, expect } from '@playwright/experimental-ct-react';
import { TerminalWindow } from './TerminalWindow';

test.describe('TerminalWindow', () => {
  test('renders with title', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow title="terminal">
        <div>Test content</div>
      </TerminalWindow>
    );
    
    await expect(component).toBeVisible();
    await expect(component.getByText('terminal')).toBeVisible();
    await expect(component.getByText('Test content')).toBeVisible();
  });

  test('has traffic light dots', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow title="terminal">
        <div>Content</div>
      </TerminalWindow>
    );
    
    await expect(component).toBeVisible();
    
    // Check for three dots by finding all rounded-full divs
    const dots = component.locator('div.rounded-full');
    await expect(dots).toHaveCount(3);
    
    // Verify each dot has a background color style (inline style from our fix)
    for (let i = 0; i < 3; i++) {
      const dot = dots.nth(i);
      await expect(dot).toBeVisible();
      
      // Check that the dot has an inline backgroundColor style
      const hasBackgroundColor = await dot.evaluate((el) => {
        const style = (el as HTMLElement).style.backgroundColor;
        return style && style !== '';
      });
      expect(hasBackgroundColor).toBe(true);
    }
  });

  test('content has monospace font', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow title="terminal">
        <div>$ rbee-keeper infer --model llama-3.1-70b</div>
      </TerminalWindow>
    );
    
    await expect(component).toBeVisible();
    await expect(component).toContainText('$ rbee-keeper infer');
    
    // The content wrapper should have font-mono class
    const contentWrapper = component.locator('div').filter({ hasText: '$ rbee-keeper' });
    await expect(contentWrapper).toBeVisible();
    
    // Check that parent has font-mono class
    const hasMonoClass = await component.evaluate((el) => {
      // Find the element with padding (content area)
      const contentArea = el.querySelector('.p-6');
      return contentArea?.classList.contains('font-mono') || false;
    });
    expect(hasMonoClass).toBe(true);
  });

  test('renders code content with proper styling', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow title="terminal">
        <div className="space-y-2">
          <div className="text-muted-foreground">
            <span className="text-chart-3">$</span> rbee-keeper infer --model llama-3.1-70b
          </div>
          <div className="text-foreground">
            <span className="text-chart-2">export</span>{' '}
            <span className="text-primary">async</span>{' '}
            <span className="text-chart-4">function</span>
          </div>
        </div>
      </TerminalWindow>
    );
    
    await expect(component).toBeVisible();
    await expect(component.getByText('export')).toBeVisible();
    await expect(component.getByText('async')).toBeVisible();
    await expect(component.getByText('function')).toBeVisible();
  });

  test('renders without title', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow>
        <div>Content without title</div>
      </TerminalWindow>
    );
    
    await expect(component).toBeVisible();
    await expect(component.getByText('Content without title')).toBeVisible();
  });

  test('applies custom className', async ({ mount }) => {
    const component = await mount(
      <TerminalWindow className="custom-class" title="test">
        <div>Content</div>
      </TerminalWindow>
    );
    
    await expect(component).toHaveClass(/custom-class/);
  });
});
