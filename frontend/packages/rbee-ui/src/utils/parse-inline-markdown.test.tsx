import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { InlineMarkdown, parseInlineMarkdown } from './parse-inline-markdown'

describe('parseInlineMarkdown', () => {
  describe('Plain text', () => {
    it('should return plain text unchanged', () => {
      const result = parseInlineMarkdown('Hello world')
      expect(result).toEqual(['Hello world'])
    })

    it('should handle empty string', () => {
      const result = parseInlineMarkdown('')
      expect(result).toEqual([''])
    })

    it('should handle text with spaces', () => {
      const result = parseInlineMarkdown('  Hello   world  ')
      expect(result).toEqual(['  Hello   world  '])
    })
  })

  describe('Bold formatting (**text**)', () => {
    it('should parse single bold word', () => {
      const result = parseInlineMarkdown('Power **your** GPUs')
      expect(result).toHaveLength(3)
      expect(result[0]).toBe('Power ')
      expect(result[2]).toBe(' GPUs')

      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('your')
    })

    it('should parse multiple bold words', () => {
      const result = parseInlineMarkdown('**First** and **second** bold')
      const { container } = render(result)
      const strongs = container.querySelectorAll('strong')
      expect(strongs).toHaveLength(2)
      expect(strongs[0]).toHaveTextContent('First')
      expect(strongs[1]).toHaveTextContent('second')
    })

    it('should parse bold at start of string', () => {
      const result = parseInlineMarkdown('**Bold** text')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('Bold')
    })

    it('should parse bold at end of string', () => {
      const result = parseInlineMarkdown('Text **bold**')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('bold')
    })

    it('should parse bold phrase with spaces', () => {
      const result = parseInlineMarkdown('This is **very important** text')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('very important')
    })

    it('should NOT parse incomplete bold markers', () => {
      const result = parseInlineMarkdown('**incomplete')
      expect(result).toEqual(['**incomplete'])
    })

    it('should NOT parse single asterisks as bold', () => {
      const result = parseInlineMarkdown('*not bold*')
      const { container } = render(result)
      expect(container.querySelector('strong')).toBeNull()
      expect(container.querySelector('em')).toHaveTextContent('not bold')
    })
  })

  describe('Italic formatting (*text*)', () => {
    it('should parse single italic word', () => {
      const result = parseInlineMarkdown('This is *italic* text')
      const { container } = render(result)
      expect(container.querySelector('em')).toHaveTextContent('italic')
    })

    it('should parse multiple italic words', () => {
      const result = parseInlineMarkdown('*First* and *second* italic')
      const { container } = render(result)
      const ems = container.querySelectorAll('em')
      expect(ems).toHaveLength(2)
      expect(ems[0]).toHaveTextContent('First')
      expect(ems[1]).toHaveTextContent('second')
    })

    it('should parse italic phrase with spaces', () => {
      const result = parseInlineMarkdown('This is *very important* text')
      const { container } = render(result)
      expect(container.querySelector('em')).toHaveTextContent('very important')
    })

    it('should NOT parse incomplete italic markers', () => {
      const result = parseInlineMarkdown('*incomplete')
      expect(result).toEqual(['*incomplete'])
    })
  })

  describe('Bold vs Italic precedence (THE BUG FIX)', () => {
    it('should parse **bold** correctly, not as *italic*', () => {
      const result = parseInlineMarkdown('Power **your** GPUs')
      const { container } = render(result)

      // Should have strong, not em
      expect(container.querySelector('strong')).toHaveTextContent('your')
      expect(container.querySelector('em')).toBeNull()

      // Should NOT have stray asterisks
      expect(container.textContent).toBe('Power your GPUs')
      expect(container.textContent).not.toContain('*')
    })

    it('should handle the exact bug case from the issue', () => {
      const text =
        'Power Zed, Cursor, and your own agents on **your** GPUs. OpenAI-compatible - drop-in, zero API fees.'
      const result = parseInlineMarkdown(text)
      const { container } = render(result)

      // Should have exactly one strong element
      const strongs = container.querySelectorAll('strong')
      expect(strongs).toHaveLength(1)
      expect(strongs[0]).toHaveTextContent('your')

      // Should have no em elements
      expect(container.querySelectorAll('em')).toHaveLength(0)

      // Should NOT contain any stray asterisks
      expect(container.textContent).not.toContain('*')
      expect(container.textContent).not.toMatch(/your\*/)
      expect(container.textContent).not.toMatch(/\*your/)
    })

    it('should handle bold and italic in same string', () => {
      const result = parseInlineMarkdown('**Bold** and *italic* text')
      const { container } = render(result)

      expect(container.querySelector('strong')).toHaveTextContent('Bold')
      expect(container.querySelector('em')).toHaveTextContent('italic')
    })

    it('should handle adjacent bold and italic', () => {
      const result = parseInlineMarkdown('**bold***italic*')
      const { container } = render(result)

      expect(container.querySelector('strong')).toHaveTextContent('bold')
      expect(container.querySelector('em')).toHaveTextContent('italic')
    })
  })

  describe('Links ([text](url))', () => {
    it('should parse simple link', () => {
      const result = parseInlineMarkdown('Read the [docs](https://example.com)')
      const { container } = render(result)
      const link = container.querySelector('a')

      expect(link).toHaveTextContent('docs')
      expect(link).toHaveAttribute('href', 'https://example.com')
    })

    it('should add target="_blank" for external links', () => {
      const result = parseInlineMarkdown('[External](https://example.com)')
      const { container } = render(result)
      const link = container.querySelector('a')

      expect(link).toHaveAttribute('target', '_blank')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    })

    it('should NOT add target="_blank" for internal links', () => {
      const result = parseInlineMarkdown('[Internal](/docs)')
      const { container } = render(result)
      const link = container.querySelector('a')

      expect(link).not.toHaveAttribute('target')
      expect(link).not.toHaveAttribute('rel')
    })

    it('should parse multiple links', () => {
      const result = parseInlineMarkdown('[First](https://one.com) and [Second](https://two.com)')
      const { container } = render(result)
      const links = container.querySelectorAll('a')

      expect(links).toHaveLength(2)
      expect(links[0]).toHaveTextContent('First')
      expect(links[0]).toHaveAttribute('href', 'https://one.com')
      expect(links[1]).toHaveTextContent('Second')
      expect(links[1]).toHaveAttribute('href', 'https://two.com')
    })

    it('should apply brand link styling', () => {
      const result = parseInlineMarkdown('[Link](https://example.com)')
      const { container } = render(result)
      const link = container.querySelector('a')

      expect(link?.className).toContain('text-[color:var(--primary)]')
      expect(link?.className).toContain('underline')
    })

    it('should NOT parse incomplete link syntax', () => {
      const result = parseInlineMarkdown('[text without url')
      expect(result).toEqual(['[text without url'])
    })

    it('should NOT parse link without brackets', () => {
      const result = parseInlineMarkdown('text(https://example.com)')
      expect(result).toEqual(['text(https://example.com)'])
    })
  })

  describe('Mixed formatting', () => {
    it('should handle bold, italic, and links together', () => {
      const text = 'Check **bold**, *italic*, and [link](https://example.com)'
      const result = parseInlineMarkdown(text)
      const { container } = render(result)

      expect(container.querySelector('strong')).toHaveTextContent('bold')
      expect(container.querySelector('em')).toHaveTextContent('italic')
      expect(container.querySelector('a')).toHaveTextContent('link')
    })

    it('should handle multiple formats in complex sentence', () => {
      const text = 'Power **your** GPUs with *zero* API fees. [Learn more](/docs)'
      const result = parseInlineMarkdown(text)
      const { container } = render(result)

      expect(container.querySelector('strong')).toHaveTextContent('your')
      expect(container.querySelector('em')).toHaveTextContent('zero')
      expect(container.querySelector('a')).toHaveTextContent('Learn more')
    })

    it('should preserve text order with mixed formatting', () => {
      const text = 'Start **bold** middle *italic* end [link](/url)'
      const result = parseInlineMarkdown(text)
      const { container } = render(result)

      expect(container.textContent).toBe('Start bold middle italic end link')
    })
  })

  describe('Edge cases', () => {
    it('should handle text with only formatting markers', () => {
      const result = parseInlineMarkdown('**bold**')
      const { container } = render(result)
      expect(container.textContent).toBe('bold')
    })

    it('should handle consecutive formatting', () => {
      const result = parseInlineMarkdown('**bold***italic***bold2**')
      const { container } = render(result)

      const strongs = container.querySelectorAll('strong')
      const ems = container.querySelectorAll('em')
      expect(strongs).toHaveLength(2)
      expect(ems).toHaveLength(1)

      // Verify the actual content
      expect(strongs[0]).toHaveTextContent('bold')
      expect(ems[0]).toHaveTextContent('italic')
      expect(strongs[1]).toHaveTextContent('bold2')
    })

    it('should handle special characters in formatted text', () => {
      const result = parseInlineMarkdown('**hello@world.com**')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('hello@world.com')
    })

    it('should handle numbers in formatted text', () => {
      const result = parseInlineMarkdown('**123** and *456*')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('123')
      expect(container.querySelector('em')).toHaveTextContent('456')
    })

    it('should handle unicode characters', () => {
      const result = parseInlineMarkdown('**你好** and *مرحبا*')
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent('你好')
      expect(container.querySelector('em')).toHaveTextContent('مرحبا')
    })

    it('should handle very long text', () => {
      const longText = 'a'.repeat(1000)
      const result = parseInlineMarkdown(`Start **${longText}** end`)
      const { container } = render(result)
      expect(container.querySelector('strong')).toHaveTextContent(longText)
    })
  })

  describe('React keys', () => {
    it('should add unique keys to React elements', () => {
      const result = parseInlineMarkdown('**one** and **two** and **three**')

      // Check that all elements have keys
      const elements = result.filter((item) => typeof item === 'object')
      elements.forEach((element) => {
        if (element && typeof element === 'object' && 'key' in element) {
          expect(element.key).toBeDefined()
        }
      })
    })
  })
})

describe('InlineMarkdown component', () => {
  it('should render plain text', () => {
    const { container } = render(<InlineMarkdown>Hello world</InlineMarkdown>)
    expect(container.textContent).toBe('Hello world')
  })

  it('should render bold text', () => {
    const { container } = render(<InlineMarkdown>Power **your** GPUs</InlineMarkdown>)
    expect(container.querySelector('strong')).toHaveTextContent('your')
  })

  it('should render italic text', () => {
    const { container } = render(<InlineMarkdown>Deploy *any* model</InlineMarkdown>)
    expect(container.querySelector('em')).toHaveTextContent('any')
  })

  it('should render links', () => {
    const { container } = render(<InlineMarkdown>[Docs](https://example.com)</InlineMarkdown>)
    const link = container.querySelector('a')
    expect(link).toHaveTextContent('Docs')
    expect(link).toHaveAttribute('href', 'https://example.com')
  })

  it('should render mixed formatting', () => {
    const { container } = render(<InlineMarkdown>**Bold**, *italic*, and [link](/url)</InlineMarkdown>)
    expect(container.querySelector('strong')).toHaveTextContent('Bold')
    expect(container.querySelector('em')).toHaveTextContent('italic')
    expect(container.querySelector('a')).toHaveTextContent('link')
  })
})
