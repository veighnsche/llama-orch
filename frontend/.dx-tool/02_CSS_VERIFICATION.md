# CSS Verification Features

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)

## Overview

The CSS verification module analyzes stylesheets and computed styles to help engineers verify their CSS changes without a browser.

## Core Features

### 1. Class Existence Check

**Problem:** "Did Tailwind generate my class?"

```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
```

**Implementation:**
1. Fetch all `<link>` and `<style>` tags
2. Download linked stylesheets
3. Parse CSS with `lightningcss`
4. Search for class selector `.cursor-pointer`
5. Report presence/absence

**Output:**
```
✓ Class 'cursor-pointer' found in stylesheet
  Location: /assets/main-abc123.css
  Definition: .cursor-pointer { cursor: pointer; }
```

### 2. Computed Style Inspection

**Problem:** "What styles are actually applied to this element?"

```bash
dx css --selector ".theme-toggle" http://localhost:3000
```

**Implementation:**
1. Parse HTML with `scraper`
2. Find element matching selector
3. Extract inline styles and class list
4. Match classes to stylesheet rules
5. Compute final styles (cascade + specificity)
6. Format output

**Output:**
```
✓ Selector: .theme-toggle
  Element: <button class="relative overflow-hidden text-muted-foreground...">
  
  Computed Styles:
    cursor: pointer
    color: rgb(148, 163, 184)
    background-color: transparent
    position: relative
    overflow: hidden
    transition: color 0.2s ease
  
  Applied Classes:
    relative → position: relative
    overflow-hidden → overflow: hidden
    text-muted-foreground → color: rgb(148, 163, 184)
    hover:text-foreground → (hover state)
```

### 3. Class List Extraction

**Problem:** "What Tailwind classes are on this element?"

```bash
dx css --list-classes --selector ".pricing-card" http://localhost:3000
```

**Output:**
```
✓ Classes on .pricing-card:
  Layout:
    - relative
    - p-8
    - space-y-6
  
  Colors:
    - bg-card
    - border-2
    - border-border
  
  Effects:
    - hover:shadow-lg
    - transition-all
  
  Custom:
    - [custom-class-name]
```

### 4. Token Verification

**Problem:** "Is this element using the correct design token?"

```bash
dx css --property "color" --selector ".nav-link" --expect "var(--muted-foreground)" http://localhost:3000
```

**Output:**
```
✓ Property 'color' on .nav-link
  Expected: var(--muted-foreground)
  Actual: var(--muted-foreground)
  Resolved: rgb(148, 163, 184)
  
  ✓ Match!
```

### 5. Unused Class Detection

**Problem:** "Are there classes in my stylesheet that aren't used?"

```bash
dx css --unused http://localhost:3000
```

**Implementation:**
1. Parse all stylesheets
2. Extract all class selectors
3. Parse HTML and find all used classes
4. Report classes in CSS but not in HTML

**Output:**
```
⚠ Unused classes detected (23 total):
  
  Tailwind utilities:
    - bg-fuchsia-600 (not used)
    - border-8 (not used)
    - rotate-2 (not used)
  
  Custom classes:
    - .old-button-style (not used)
    - .deprecated-card (not used)
  
  Suggestion: These classes may be safe to remove or are tree-shaken in production.
```

### 6. Missing Class Detection

**Problem:** "Why isn't my class working?"

```bash
dx css --selector ".theme-toggle" --expect-class "cursor-pointer" http://localhost:3000
```

**Output:**
```
✗ Class 'cursor-pointer' not found on .theme-toggle
  
  Element classes:
    relative overflow-hidden text-muted-foreground hover:text-foreground
  
  Possible causes:
    1. Class not added to element
    2. Class removed by JavaScript
    3. Conditional rendering issue
  
  Stylesheet check:
    ✓ Class 'cursor-pointer' exists in stylesheet
    → Class is available but not applied to element
```

### 7. Specificity Analysis

**Problem:** "Why is my style being overridden?"

```bash
dx css --specificity --selector ".theme-toggle" --property "cursor" http://localhost:3000
```

**Output:**
```
✓ Specificity analysis for 'cursor' on .theme-toggle
  
  Applied Rules (in order of specificity):
    1. .cursor-pointer { cursor: pointer; }
       Specificity: (0,1,0)
       Source: /assets/main.css:1234
       Status: ✓ APPLIED
  
    2. button { cursor: default; }
       Specificity: (0,0,1)
       Source: user-agent stylesheet
       Status: ✗ OVERRIDDEN
  
  Final Value: pointer
```

### 8. Responsive Breakpoint Testing

**Problem:** "What styles apply at different screen sizes?"

```bash
dx css --selector ".nav" --breakpoint "mobile" http://localhost:3000
```

**Output:**
```
✓ Styles for .nav at breakpoint 'mobile' (max-width: 768px)
  
  Base styles:
    display: flex
    gap: 2rem
  
  Mobile-specific (@media):
    display: none
    
  Active classes:
    hidden md:flex → display: none (mobile), display: flex (desktop)
```

## Advanced Features

### 9. CSS Variable Resolution

```bash
dx css --resolve-vars --selector ".theme-toggle" http://localhost:3000
```

**Output:**
```
✓ CSS Variables on .theme-toggle
  
  --muted-foreground: #94a3b8
    ↳ Defined in: :root
    ↳ Used in: color property
  
  --foreground: #f1f5f9
    ↳ Defined in: :root
    ↳ Used in: hover:text-foreground
```

### 10. Dark Mode Verification

```bash
dx css --dark-mode --selector ".theme-toggle" http://localhost:3000
```

**Output:**
```
✓ Dark mode styles for .theme-toggle
  
  Light mode:
    color: rgb(100, 116, 139)
  
  Dark mode (.dark):
    color: rgb(148, 163, 184)
  
  Difference: +48 lightness
```

## Implementation Details

### CSS Parser

```rust
use lightningcss::{stylesheet::StyleSheet, printer::PrinterOptions};

pub fn parse_stylesheet(css: &str) -> Result<StyleSheet> {
    StyleSheet::parse(css, ParserOptions::default())
}

pub fn find_class(stylesheet: &StyleSheet, class_name: &str) -> Option<Rule> {
    // Search for .class-name in rules
}
```

### Style Computation

```rust
pub struct ComputedStyle {
    selector: String,
    properties: HashMap<String, String>,
    sources: HashMap<String, RuleSource>,
}

pub fn compute_styles(html: &Html, selector: &str) -> Result<ComputedStyle> {
    // 1. Find element
    // 2. Extract classes
    // 3. Match to stylesheet rules
    // 4. Apply cascade and specificity
    // 5. Resolve CSS variables
}
```

### Specificity Calculator

```rust
pub struct Specificity {
    inline: u32,
    ids: u32,
    classes: u32,
    elements: u32,
}

impl Specificity {
    pub fn calculate(selector: &str) -> Self {
        // Parse selector and count components
    }
    
    pub fn compare(&self, other: &Self) -> Ordering {
        // Compare specificity values
    }
}
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_class_exists() {
        let css = ".cursor-pointer { cursor: pointer; }";
        assert!(find_class(css, "cursor-pointer").is_some());
    }
    
    #[test]
    fn test_computed_style() {
        let html = r#"<button class="cursor-pointer">Click</button>"#;
        let css = ".cursor-pointer { cursor: pointer; }";
        let style = compute_styles(html, "button").unwrap();
        assert_eq!(style.get("cursor"), Some("pointer"));
    }
}
```

---

**Next:** See `03_HTML_QUERIES.md` for HTML structure querying features.
