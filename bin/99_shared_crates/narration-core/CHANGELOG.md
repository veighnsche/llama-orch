# Changelog

All notable changes to narration-core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-10-21 — TEAM-191 Format & Factory 🏭

### Breaking Changes ⚠️
- **Output Format Changed**: Actor now appears inline on same line (not on separate line)
  - **Old**: `[actor]\n  message`
  - **New**: `[actor                ] message` (20-char column alignment)
  - **Impact**: Better readability, consistent column alignment for messages
  - **Migration**: No code changes needed, only output format changes

### Added
- 🏭 **NarrationFactory** - Define default actor once per crate, reuse everywhere
  - `const fn new(actor: &'static str)` - Create factory
  - `.narrate(action, target)` - Create narration with default actor
  - `.actor()` - Get factory's default actor
  - Reduces boilerplate significantly
  - Compile-time constant support
- 📏 **Column Alignment** - Actor field padded to 20 chars for consistent message alignment
- ✨ **Improved Readability** - Messages start at same column, easier to scan logs

### Improved
- **Log Scanning** - Consistent column alignment makes it easier to scan through logs
- **Crate-level Consistency** - Factory pattern ensures same actor used throughout crate

### Example
```rust
// Define at module/crate level
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

// Use throughout the crate
NARRATE.narrate(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

---

## [0.3.0] - 2025-10-21 — TEAM-191 Upgrade 🎀

### Added
- 📊 **Table Formatting Documentation** - Comprehensive guide for `.table()` method with examples
- 👑 **Queen-Rbee Actor Constants** - `ACTOR_QUEEN_RBEE`, `ACTOR_QUEEN_ROUTER` (with cute emojis!)
- 🎯 **Job Routing Action Constants** - `ACTION_ROUTE_JOB`, `ACTION_PARSE_OPERATION`, `ACTION_JOB_CREATE`
- 🏠 **Hive Management Action Constants** - `ACTION_HIVE_INSTALL`, `ACTION_HIVE_UNINSTALL`, `ACTION_HIVE_START`, `ACTION_HIVE_STOP`, `ACTION_HIVE_STATUS`, `ACTION_HIVE_LIST`
- 🚀 **System Action Constants** - `ACTION_STATUS`, `ACTION_START`, `ACTION_LISTEN`, `ACTION_READY`, `ACTION_ERROR`
- 😊 **Emoji Support Validation** - Confirmed emojis work perfectly in human fields
- 📝 **Multi-line Message Support** - Long formatted messages with newlines fully supported

### Improved
- 📚 **README** - Table formatting prominently documented with examples and output
- 🎯 **Taxonomy** - Extended with 15+ new action constants for queen-rbee operations
- 🎀 **Editorial Quality** - Queen-rbee usage patterns reviewed and approved (⭐⭐⭐⭐⭐)

### Fixed
- None! Everything was already working! 🎉

### What We Learned
- Table formatting is AMAZING for status displays and lists!
- Emojis make debugging delightful! Keep using them! 🎀
- Multi-line messages work great for complex operations!
- Queen-rbee team writes excellent narrations! 💝

## [0.2.0] - 2025-10-04

### Added
- **Builder Pattern API** - Ergonomic fluent API reduces boilerplate by 43%
- **Axum Middleware** - Built-in correlation ID middleware with auto-extraction/generation
- **Comprehensive Documentation** - Policy guide, field reference, troubleshooting sections
- **Code Quality** - Reduced duplication from ~400 lines to ~90 lines (78% reduction)

### Testing & Quality
- **100% Functional Test Pass Rate** - 75/75 tests passing (50 unit + 16 integration + 9 property)
- **Zero Flaky Tests** - Fixed global state issues with improved `CaptureAdapter`
- **Property-Based Tests** - Comprehensive invariant testing for security & correctness
- **Comprehensive Specification** - 42 normative requirements (NARR-1001..NARR-8005)

### Features
- **7 Logging Levels** - MUTE, TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- **6 Secret Patterns** - Bearer tokens, API keys, JWT, private keys, URL passwords, UUIDs
- **Auto-Injection** - `narrate_auto()` automatically adds provenance metadata
- **Correlation ID Helpers** - Generate, validate (<100ns), extract from headers
- **HTTP Context Propagation** - Extract/inject correlation IDs from HTTP headers
- **Unicode Safety** - ASCII fast path, CRLF sanitization, zero-width character filtering
- **ReDoS-Safe Redaction** - Bounded quantifiers with `OnceLock` caching

## [0.1.0] - 2025-09-15

### Added
- Initial release
- Basic narration functionality with actor/action/target taxonomy
- Secret redaction (Bearer tokens, API keys)
- Correlation ID support
- Test capture adapter for BDD tests
- Human-readable descriptions
- Cute mode (children's book narration) 🎀
- Story mode (dialogue-based narration) 🎭

### Core Features
- Structured logging with `tracing` backend
- Automatic secret redaction
- Correlation ID propagation
- Test capture adapter
- OpenTelemetry integration (optional)

---

**Maintained by**: The Narration Core Team 🎀  
**License**: GPL-3.0-or-later

*May your logs be readable, your correlation IDs present, and your debugging experience absolutely DELIGHTFUL! 🎀✨*
