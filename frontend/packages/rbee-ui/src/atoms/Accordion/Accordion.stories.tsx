// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './Accordion'

const meta: Meta<typeof Accordion> = {
  title: 'Atoms/Accordion',
  component: Accordion,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Accordion>

/**
 * ## Overview
 * Accordion displays collapsible content panels. Built on Radix UI Accordion primitive
 * with smooth animations and keyboard navigation support.
 *
 * ## Composition
 * - Accordion (container)
 * - AccordionItem (individual panel)
 * - AccordionTrigger (clickable header)
 * - AccordionContent (collapsible content)
 *
 * ## When to Use
 * - FAQ sections
 * - Progressive disclosure
 * - Reduce page length
 * - Group related information
 *
 * ## Used In
 * - FAQSection (primary use case)
 */

export const Default: Story = {
  render: () => (
    <Accordion type="single" collapsible className="w-[500px]">
      <AccordionItem value="item-1">
        <AccordionTrigger>What is rbee?</AccordionTrigger>
        <AccordionContent>
          rbee is a private LLM hosting platform based in the Netherlands. We make it easy to deploy and manage
          open-source language models with enterprise-grade security and compliance.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2">
        <AccordionTrigger>How does pricing work?</AccordionTrigger>
        <AccordionContent>
          We offer flexible pricing based on your usage. Choose from our Starter, Professional, or Enterprise plans.
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-3">
        <AccordionTrigger>Is my data secure?</AccordionTrigger>
        <AccordionContent>
          Yes. All data is encrypted in transit and at rest. We are SOC 2 compliant and follow GDPR regulations.
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  ),
}

export const Multiple: Story = {
  render: () => (
    <Accordion type="multiple" className="w-[500px]">
      <AccordionItem value="item-1">
        <AccordionTrigger>Getting Started</AccordionTrigger>
        <AccordionContent>
          <ol className="list-decimal list-inside space-y-1 text-sm">
            <li>Create an account</li>
            <li>Choose a model</li>
            <li>Deploy with one click</li>
          </ol>
        </AccordionContent>
      </AccordionItem>
      <AccordionItem value="item-2">
        <AccordionTrigger>Supported Models</AccordionTrigger>
        <AccordionContent>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>LLaMA 3 (8B, 70B)</li>
            <li>Mistral (7B, Mixtral 8x7B)</li>
            <li>Qwen 2.5</li>
          </ul>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  ),
}

export const Collapsible: Story = {
  render: () => (
    <Accordion type="single" collapsible defaultValue="item-1" className="w-[500px]">
      <AccordionItem value="item-1">
        <AccordionTrigger>Technical Specifications</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>GPU Options</span>
              <span className="font-medium">A100, H100</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Memory</span>
              <span className="font-medium">Up to 80GB VRAM</span>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  ),
}

export const InFAQContext: Story = {
  render: () => (
    <div className="max-w-3xl mx-auto py-12">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold mb-3">Frequently Asked Questions</h2>
        <p className="text-muted-foreground">Everything you need to know</p>
      </div>
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="what-is">
          <AccordionTrigger>What is rbee?</AccordionTrigger>
          <AccordionContent>
            rbee is a private LLM hosting platform that allows you to deploy and manage open-source language models
            with enterprise-grade security.
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="pricing">
          <AccordionTrigger>How does pricing work?</AccordionTrigger>
          <AccordionContent>
            We offer three pricing tiers: Starter (€29/month), Professional (€99/month), and Enterprise (Custom).
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="security">
          <AccordionTrigger>Is my data secure?</AccordionTrigger>
          <AccordionContent>
            Yes. All data is encrypted, we are SOC 2 certified, and GDPR compliant. Your data never leaves the EU.
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  ),
}
