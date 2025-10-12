# Frontend DX Tool - Master Plan

**Created by:** TEAM-FE-011 (aka TEAM-DX-000)  
**Status:** Planning Phase  
**Priority:** P1 - Improves developer experience significantly

## Problem Statement

Frontend engineers working without browser access (SSH, remote environments, CI/CD) currently:
- ❌ Must manually curl/wget and parse HTML/CSS output
- ❌ Create ad-hoc bash scripts that are complex and unmaintainable
- ❌ Cannot easily verify visual changes without manual browser inspection
- ❌ Waste time sifting through raw HTML/CSS dumps

## Solution: Frontend DX CLI Tool

A robust CLI tool that provides **browser-like verification** for frontend changes without requiring a browser.

### Core Capabilities

1. **CSS Verification**
   - Extract computed styles for specific selectors
   - Verify Tailwind classes are generated
   - Check color tokens, fonts, spacing
   - Detect missing/unused classes

2. **HTML Structure Verification**
   - Query DOM structure (like browser DevTools)
   - Extract element attributes, classes, text content
   - Verify component rendering
   - Check accessibility attributes

3. **Visual Regression Detection**
   - Compare rendered output between versions
   - Detect layout shifts, color changes
   - Generate diff reports

4. **Component Testing**
   - Test individual components in isolation
   - Verify props/state rendering
   - Check responsive behavior

## Technology Stack

**Language:** Rust (performance, reliability, single binary distribution)

**Core Libraries:**
- `reqwest` - HTTP client for fetching pages
- `scraper` - HTML parsing and CSS selector queries
- `lightningcss` - CSS parsing and analysis
- `clap` - CLI argument parsing
- `serde` - JSON serialization for reports
- `tokio` - Async runtime
- `colored` - Terminal output formatting
- `insta` - Snapshot testing

**Why Rust:**
- Single binary distribution (no runtime dependencies)
- Fast execution (important for CI/CD)
- Strong type safety (fewer bugs)
- Excellent HTML/CSS parsing libraries
- Cross-platform (Linux, macOS, Windows)

## Architecture

```
dx-tool/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── commands/            # Command implementations
│   │   ├── css.rs           # CSS verification commands
│   │   ├── html.rs          # HTML query commands
│   │   ├── snapshot.rs      # Visual regression commands
│   │   └── component.rs     # Component testing commands
│   ├── fetcher/             # HTTP fetching logic
│   ├── parser/              # HTML/CSS parsing
│   ├── analyzer/            # Style computation, diff detection
│   ├── reporter/            # Output formatting
│   └── config/              # Configuration management
├── tests/                   # Integration tests
├── examples/                # Usage examples
└── Cargo.toml               # Dependencies
```

## Deliverables

1. **CLI Tool Binary** - `dx` command
2. **Documentation** - Usage guide, examples
3. **Integration** - pnpm scripts, CI/CD integration
4. **Test Suite** - Comprehensive tests

## Success Criteria

✅ Engineers can verify CSS changes without browser  
✅ Tool runs in <2 seconds for typical queries  
✅ Clear, actionable output (not raw HTML dumps)  
✅ Works in CI/CD pipelines  
✅ Single binary, no runtime dependencies  
✅ Comprehensive documentation with examples

## Implementation Phases

See individual planning documents:
- `01_CLI_DESIGN.md` - Command structure and UX
- `02_CSS_VERIFICATION.md` - CSS analysis features
- `03_HTML_QUERIES.md` - HTML querying features
- `04_VISUAL_REGRESSION.md` - Snapshot testing
- `05_INTEGRATION.md` - Workspace integration
- `06_IMPLEMENTATION_ROADMAP.md` - Development timeline

## Related Issues

- Solves the Tailwind scanning issue indirectly (can verify generated classes)
- Enables automated visual regression testing
- Improves CI/CD feedback loop

---

**Next Steps:** Review individual planning documents, approve architecture, assign to implementation team.
