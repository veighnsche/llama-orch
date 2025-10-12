# TEAM-DX-001 KICKOFF

**From:** TEAM-FE-011 (aka TEAM-DX-000)  
**To:** TEAM-DX-001  
**Date:** 2025-10-12  
**Status:** Ready to Start  
**Priority:** P1 - Critical DX Infrastructure

---

## Mission

Build the **Frontend DX CLI Tool** - a Rust-based command-line tool that allows frontend engineers to verify CSS and HTML changes without browser access.

## Why This Matters

Frontend engineers currently:
- ❌ Waste hours manually curling pages and parsing HTML/CSS
- ❌ Create fragile bash scripts that break constantly
- ❌ Cannot verify visual changes without manual browser inspection
- ❌ Have poor DX in SSH/remote/CI environments

**This tool solves all of that.**

## What You're Building

A single-binary CLI tool (`dx`) that provides:

1. **CSS Verification** - Check if Tailwind classes exist, inspect computed styles
2. **HTML Queries** - DOM inspection like browser DevTools
3. **Visual Regression** - Snapshot testing for structural/style changes
4. **Component Testing** - Test Vue components in isolation

**Example Usage:**
```bash
# Check if a class was generated
dx css --class-exists "cursor-pointer" http://localhost:3000

# Inspect element styles
dx css --selector ".theme-toggle" http://localhost:3000

# Query DOM structure
dx html --selector "nav" --tree http://localhost:3000

# Create snapshot baseline
dx snapshot --create --name "homepage" http://localhost:3000
```

## Your Resources

### Planning Documents (READ THESE FIRST)

Located in `frontend/.dx-tool/`:

1. **`00_MASTER_PLAN.md`** - Overall vision, tech stack, success criteria
2. **`01_CLI_DESIGN.md`** - Complete CLI UX with examples
3. **`02_CSS_VERIFICATION.md`** - CSS analysis features
4. **`03_HTML_QUERIES.md`** - HTML querying features
5. **`04_VISUAL_REGRESSION.md`** - Snapshot testing
6. **`05_INTEGRATION.md`** - Workspace integration
7. **`06_IMPLEMENTATION_ROADMAP.md`** - 5-week timeline

### Technology Stack

**Language:** Rust (for performance, reliability, single binary)

**Core Dependencies:**
```toml
[dependencies]
reqwest = "0.11"        # HTTP client
scraper = "0.18"        # HTML parsing
lightningcss = "1.0"    # CSS parsing
clap = "4.4"            # CLI framework
serde = "1.0"           # JSON serialization
tokio = "1.35"          # Async runtime
colored = "2.1"         # Terminal colors
insta = "1.34"          # Snapshot testing
```

## Phase 1: Foundation (Week 1)

Your immediate tasks:

### 1. Project Setup

```bash
cd frontend/.dx-tool
cargo init --name dx
```

Create this structure:
```
frontend/.dx-tool/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── commands/            # Command implementations
│   │   ├── mod.rs
│   │   ├── css.rs
│   │   ├── html.rs
│   │   └── snapshot.rs
│   ├── fetcher/             # HTTP fetching
│   │   ├── mod.rs
│   │   └── client.rs
│   ├── parser/              # HTML/CSS parsing
│   │   ├── mod.rs
│   │   ├── html.rs
│   │   └── css.rs
│   └── lib.rs
├── tests/                   # Integration tests
├── Cargo.toml
└── README.md
```

### 2. Implement HTTP Fetcher

```rust
// src/fetcher/client.rs
use reqwest;

pub async fn fetch_page(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}
```

### 3. Implement HTML Parser

```rust
// src/parser/html.rs
use scraper::{Html, Selector};

pub fn parse_html(html: &str) -> Html {
    Html::parse_document(html)
}

pub fn select_elements(html: &Html, selector: &str) -> Result<Vec<ElementRef>> {
    let selector = Selector::parse(selector)?;
    Ok(html.select(&selector).collect())
}
```

### 4. Basic CLI Structure

```rust
// src/main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "dx")]
#[command(about = "Frontend DX CLI Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// CSS verification commands
    Css {
        #[arg(long)]
        class_exists: Option<String>,
        
        #[arg(long)]
        selector: Option<String>,
        
        url: String,
    },
    /// HTML query commands
    Html {
        #[arg(long)]
        selector: String,
        
        #[arg(long)]
        attrs: bool,
        
        url: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Css { class_exists, selector, url } => {
            // Implement CSS commands
        }
        Commands::Html { selector, attrs, url } => {
            // Implement HTML commands
        }
    }
}
```

### 5. First Working Command

Implement `dx css --class-exists`:

```rust
// src/commands/css.rs
pub async fn check_class_exists(url: &str, class_name: &str) -> Result<bool> {
    let html = fetch_page(url).await?;
    let stylesheets = extract_stylesheets(&html).await?;
    
    for css in stylesheets {
        if css.contains(&format!(".{}", class_name)) {
            return Ok(true);
        }
    }
    
    Ok(false)
}
```

## Acceptance Criteria for Phase 1

By end of Week 1, you must have:

✅ Rust project initialized with proper structure  
✅ HTTP fetcher working (can fetch localhost:3000)  
✅ HTML parser working (can parse and select elements)  
✅ Basic CLI with `clap` (can parse arguments)  
✅ **ONE working command:** `dx css --class-exists "cursor-pointer" http://localhost:3000`  
✅ Unit tests for fetcher and parser  
✅ Binary compiles and runs  

## Testing Your Work

```bash
# Start the commercial frontend
cd frontend/bin/commercial
pnpm dev

# In another terminal, test your tool
cd frontend/.dx-tool
cargo run -- css --class-exists "cursor-pointer" http://localhost:3000

# Expected output:
# ✓ Class 'cursor-pointer' found in stylesheet
```

## Success Metrics

- ✅ Command runs in <2 seconds
- ✅ Clear, actionable output (not raw dumps)
- ✅ Proper error handling with helpful messages
- ✅ Code follows Rust best practices
- ✅ 80%+ test coverage

## Engineering Rules

**MANDATORY:** Read `.windsurf/rules/engineering-rules.md` before starting.

Key rules for this project:
- ✅ Add TEAM-DX-001 signatures to new files
- ✅ Write tests for all core functionality
- ✅ Document your code with examples
- ✅ No TODO markers - implement or ask for help
- ✅ Handoff must be ≤2 pages with code examples

## Resources

**Rust Documentation:**
- Rust Book: https://doc.rust-lang.org/book/
- Clap docs: https://docs.rs/clap/
- Scraper docs: https://docs.rs/scraper/
- Reqwest docs: https://docs.rs/reqwest/

**Project Context:**
- This tool solves the Tailwind scanning issue (TEAM-FE-012's problem)
- Will be used in CI/CD pipelines
- Must work on Linux (primary target)

## Communication

**Questions?** Ask in your handoff document. Don't guess.

**Blockers?** Document them clearly with:
1. What you tried
2. What failed
3. What you need to proceed

## Next Steps

1. **Read all planning documents** (30 minutes)
2. **Read engineering rules** (10 minutes)
3. **Set up Rust project** (1 hour)
4. **Implement HTTP fetcher** (2 hours)
5. **Implement HTML parser** (2 hours)
6. **Build basic CLI** (3 hours)
7. **Implement first command** (4 hours)
8. **Write tests** (2 hours)
9. **Document your work** (1 hour)

**Total estimated time:** ~15 hours for Phase 1

## Handoff Requirements

When you hand off to TEAM-DX-002, include:

1. **What you built** - List of implemented features with code examples
2. **How to use it** - Commands that work, with examples
3. **What's next** - Clear priorities for Phase 2 (from roadmap)
4. **Blockers** - Any issues that need resolution
5. **Tests** - Test coverage report

**Keep it ≤2 pages.**

---

## The Bottom Line

You're building critical infrastructure that will:
- Save frontend engineers hours every week
- Enable automated visual regression testing
- Improve CI/CD feedback loops
- Solve the Tailwind scanning verification problem

**This is high-impact work. Make it count.**

**TEAM-DX-000 OUT. GOOD LUCK TEAM-DX-001. READ THE DOCS. BUILD SOMETHING GREAT.**

---

**Checklist Before You Start:**
- [ ] Read all 7 planning documents
- [ ] Read engineering rules
- [ ] Understand the problem we're solving
- [ ] Have Rust installed and working
- [ ] Can run `cargo --version`
- [ ] Commercial frontend runs on localhost:3000
- [ ] Ready to write Rust code

**Once all boxes are checked, BEGIN PHASE 1.**
