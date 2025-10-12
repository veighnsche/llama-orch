# HTML Structure Queries

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## Overview

The HTML query module provides browser DevTools-like DOM inspection capabilities from the command line.

## Core Features

### 1. Element Selection

**Problem:** "Does this element exist?"

```bash
dx html --selector ".theme-toggle" http://localhost:3000
```

**Output:**
```
✓ Found 1 element matching '.theme-toggle'
  Tag: button
  Classes: relative overflow-hidden text-muted-foreground hover:text-foreground
  Attributes: aria-label="Toggle theme" type="button"
```

### 2. Attribute Extraction

**Problem:** "What attributes does this element have?"

```bash
dx html --selector ".theme-toggle" --attrs http://localhost:3000
```

**Output:**
```
✓ Attributes for .theme-toggle
  aria-label: "Toggle theme"
  type: "button"
  class: "relative overflow-hidden text-muted-foreground..."
  data-slot: "button"
```

### 3. Text Content Extraction

**Problem:** "What text is rendered?"

```bash
dx html --selector "h1" --text http://localhost:3000
```

**Output:**
```
✓ Text content for h1
  "Welcome to rbee"
  
  Trimmed: "Welcome to rbee"
  Length: 15 characters
```

### 4. Element Counting

**Problem:** "How many buttons are on the page?"

```bash
dx html --selector "button" --count http://localhost:3000
```

**Output:**
```
✓ Found 8 elements matching 'button'
  
  Breakdown by class:
    - .theme-toggle: 1
    - .mobile-menu-toggle: 1
    - [variant="default"]: 2
    - [variant="outline"]: 4
```

### 5. DOM Tree Visualization

**Problem:** "What's the structure around this element?"

```bash
dx html --selector "nav" --tree http://localhost:3000
```

**Output:**
```
✓ DOM tree for nav
  
  nav.fixed.top-0.left-0.right-0.z-50
  ├── div.max-w-7xl.mx-auto.px-4
  │   ├── div.flex.items-center.justify-between.h-16
  │   │   ├── a.flex.items-center.gap-2 (logo)
  │   │   ├── div.hidden.md:flex.items-center.gap-8
  │   │   │   ├── a (Features)
  │   │   │   ├── a (Use Cases)
  │   │   │   ├── a (Pricing)
  │   │   │   ├── button.theme-toggle
  │   │   │   └── button (Join Waitlist)
  │   │   └── button.md:hidden (mobile menu)
```

### 6. Accessibility Audit

**Problem:** "Is this element accessible?"

```bash
dx html --selector "button" --a11y http://localhost:3000
```

**Output:**
```
✓ Accessibility audit for button.theme-toggle
  
  ✓ Has accessible name (aria-label: "Toggle theme")
  ✓ Has valid role (button)
  ✓ Is keyboard accessible (native button element)
  ⚠ Missing focus indicator (consider adding focus-visible styles)
  ✓ Has sufficient color contrast (4.8:1)
  
  Score: 4/5 (Good)
  
  Recommendations:
    - Add visible focus indicator for keyboard navigation
```

### 7. Component Detection

**Problem:** "What Vue components are rendered?"

```bash
dx html --components http://localhost:3000
```

**Output:**
```
✓ Detected components
  
  Navigation (1 instance)
    Location: nav
    Props: (inferred from attributes)
  
  ThemeToggle (1 instance)
    Location: nav button.theme-toggle
    Props: size="icon"
  
  Button (8 instances)
    Locations: nav, .hero-section, .pricing-section
    Variants: default (2), outline (4), ghost (2)
```

### 8. Form Analysis

**Problem:** "What form fields are on the page?"

```bash
dx html --forms http://localhost:3000
```

**Output:**
```
✓ Found 1 form
  
  Form #1 (Email capture)
    Action: /api/waitlist
    Method: POST
    
    Fields:
      - email (type: email, required: true)
        Label: "Email address"
        Validation: pattern="[^@]+@[^@]+\.[^@]+"
      
      - consent (type: checkbox, required: true)
        Label: "I agree to receive updates"
    
    Submit button: "Join Waitlist"
```

### 9. Link Extraction

**Problem:** "What links are on the page?"

```bash
dx html --links http://localhost:3000
```

**Output:**
```
✓ Found 15 links
  
  Internal links (12):
    - / (Home)
    - /features
    - /use-cases
    - /pricing
    - /developers
    - /providers
    - /enterprise
    ...
  
  External links (3):
    - https://github.com/veighnsche/llama-orch
    - https://twitter.com/rbee_app
    - https://discord.gg/rbee
  
  Broken links: 0
```

### 10. Meta Tag Extraction

**Problem:** "Are SEO tags correct?"

```bash
dx html --meta http://localhost:3000
```

**Output:**
```
✓ Meta tags
  
  Basic:
    <title>rbee - Build with AI. Own Your Infrastructure.</title>
    <meta name="description" content="Open-source AI orchestration...">
  
  Open Graph:
    og:title: "rbee - Build with AI. Own Your Infrastructure."
    og:description: "Open-source AI orchestration platform..."
    og:image: "/og-image.png"
    og:type: "website"
  
  Twitter Card:
    twitter:card: "summary_large_image"
    twitter:title: "rbee - Build with AI..."
    twitter:image: "/og-image.png"
  
  ✓ All required meta tags present
```

## Advanced Features

### 11. XPath Queries

```bash
dx html --xpath "//button[@aria-label='Toggle theme']" http://localhost:3000
```

### 12. Data Attribute Extraction

```bash
dx html --data-attrs --selector "[data-testid]" http://localhost:3000
```

**Output:**
```
✓ Elements with data attributes
  
  data-testid="theme-toggle"
    Element: button.theme-toggle
    Location: nav
  
  data-testid="mobile-menu"
    Element: div.mobile-menu
    Location: nav
```

### 13. Event Handler Detection

```bash
dx html --events --selector "button" http://localhost:3000
```

**Output:**
```
✓ Event handlers on button elements
  
  button.theme-toggle
    @click: "toggleDark()"
    Detected: Vue event handler
  
  button.mobile-menu-toggle
    @click: "toggleMobileMenu"
    Detected: Vue event handler
```

### 14. Conditional Rendering Check

```bash
dx html --conditional --selector ".mobile-menu" http://localhost:3000
```

**Output:**
```
✓ Conditional rendering for .mobile-menu
  
  Condition: v-if="mobileMenuOpen"
  Current state: false (element not rendered)
  
  To test true state:
    dx html --selector ".mobile-menu" --with-state '{"mobileMenuOpen":true}'
```

## Implementation Details

### HTML Parser

```rust
use scraper::{Html, Selector};

pub fn parse_html(html: &str) -> Html {
    Html::parse_document(html)
}

pub fn select(html: &Html, selector: &str) -> Result<Vec<ElementRef>> {
    let selector = Selector::parse(selector)
        .map_err(|e| Error::InvalidSelector(e))?;
    Ok(html.select(&selector).collect())
}
```

### Attribute Extraction

```rust
pub fn extract_attributes(element: &ElementRef) -> HashMap<String, String> {
    element.value()
        .attrs()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}
```

### Accessibility Checker

```rust
pub struct A11yAudit {
    has_accessible_name: bool,
    has_valid_role: bool,
    is_keyboard_accessible: bool,
    has_focus_indicator: bool,
    color_contrast: Option<f32>,
}

pub fn audit_accessibility(element: &ElementRef, styles: &ComputedStyle) -> A11yAudit {
    // Check ARIA attributes, roles, keyboard accessibility, contrast
}
```

### Tree Builder

```rust
pub struct DomTree {
    tag: String,
    classes: Vec<String>,
    children: Vec<DomTree>,
}

pub fn build_tree(element: &ElementRef, depth: usize) -> DomTree {
    // Recursively build tree structure
}

pub fn format_tree(tree: &DomTree) -> String {
    // Format with box-drawing characters
}
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_element_selection() {
        let html = r#"<button class="theme-toggle">Toggle</button>"#;
        let doc = Html::parse_document(html);
        let elements = select(&doc, ".theme-toggle").unwrap();
        assert_eq!(elements.len(), 1);
    }
    
    #[test]
    fn test_attribute_extraction() {
        let html = r#"<button aria-label="Toggle theme">Toggle</button>"#;
        let doc = Html::parse_document(html);
        let element = select(&doc, "button").unwrap()[0];
        let attrs = extract_attributes(&element);
        assert_eq!(attrs.get("aria-label"), Some(&"Toggle theme".to_string()));
    }
}
```

---

**Next:** See `04_VISUAL_REGRESSION.md` for snapshot testing features.
