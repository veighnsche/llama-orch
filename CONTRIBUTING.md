# Contributing

Thank you for your interest in contributing!

- This repository follows a contract-first, TDD workflow.
- Do not implement business logic before contracts, stubs, and tests are in place.
- Run:
  - `cargo fmt --all`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo build --workspace`
  - `cargo test --workspace --all-features -- --nocapture`

## Xtask

Use `cargo xtask` subcommands to regenerate contracts and run CI helpers.

## AI

*knock knock* human here:

I encourage vibe coders. But I still hold a vibe coding standard. Don't just ask ChatGPT to spot performance issues. Then make some random PR filled with random AI generated text (That I will AI summurize further hehe), BUT I really encourage you to use ChatGPT or Windsurf ot any other AI tool. To explain to you WHYYYYY it might be a "good" PR. Have you asked the AI to run all the tests and to see if it broke anything? Has your AI added any new tests? did your AI change any behavior that is needed upstream? Eeeh.. actually on second thouhgt. why not add an AI in the ci that rates the PR for time-wastery. (haha 😂, I'm already imaginng how much fun it would be to mess with that time-wastery AI). uuuh. So no.. I'm not going to add an AI in the ci that rates the PR for time-wastery. You just have to be nice for now. Thanks

And if you are a good contributor. YOU  ARE  THE  BEST  !
