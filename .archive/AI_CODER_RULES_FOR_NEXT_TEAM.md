# AI CODER RULES - MANDATORY READING

**Written after TEAM-328 wasted $4.00 and 30 minutes on a 10-minute task**

---

## RULE 1: READ THE ACTUAL WORDS

**User says:** "Can rebuild call uninstall and install?"  
**You hear:** Make rebuild USE those functions (composition)  
**NOT:** Delete everything and merge into one function

### How to Read
1. Read the user's message word-by-word
2. Don't interpret
3. Don't assume
4. Don't guess what they "really mean"

### Example
**User:** "Can we consolidate these functions?"  
**WRONG:** Delete all functions, create one mega-function  
**RIGHT:** Make functions call each other to avoid duplication

---

## RULE 2: ASK QUESTIONS WHEN UNCLEAR

**If you're not 100% sure what the user wants, ASK.**

### Questions to Ask
- "Do you want to keep all 3 operations or merge them?"
- "Should install build automatically if binary not found?"
- "Is the goal to reuse code or merge operations?"

### NEVER Guess
- Guessing wastes time
- Guessing wastes money
- Guessing pisses off the user

**Cost of asking:** 1 message ($0.05)  
**Cost of guessing wrong:** 20+ messages ($1.00+)

---

## RULE 3: UNDERSTAND BEFORE TOUCHING CODE

**Before you change ANYTHING:**

1. **Read the existing code** - All of it
2. **Understand what it does** - Actually understand
3. **Identify the actual problem** - Not what you think it is
4. **Think of the solution** - Before touching code
5. **Verify your understanding** - Ask if unsure

### NEVER
- Delete code without understanding it
- Create new functions without checking existing ones
- Assume you know better than the existing code

---

## RULE 4: THINK ABOUT USER EXPERIENCE

**Before implementing, ask:**

- "How will a user actually use this?"
- "What if they're on a fresh clone?"
- "What's the most intuitive behavior?"
- "What edge cases exist?"

### Example: install_to_local_bin()

**Bad UX:**
```rust
// User must build manually first
fn install_to_local_bin() {
    let binary = find_binary()?; // Fails if not built
    copy(binary)?;
}
```

**Good UX:**
```rust
// Builds automatically if needed
fn install_to_local_bin() {
    let binary = match find_binary() {
        Ok(path) => path,
        Err(_) => build()?, // Auto-build
    };
    copy(binary)?;
}
```

**Think:** "What if nothing is built yet?"

---

## RULE 5: DRY = COMPOSITION, NOT DELETION

**DRY (Don't Repeat Yourself) means:**
- Make functions call each other
- Extract common logic into shared functions
- Reuse code through composition

**DRY does NOT mean:**
- Delete everything
- Merge all functions into one
- Remove features

### Example

**WRONG (Deletion):**
```rust
// Delete install, uninstall, rebuild
// Create one mega-function
fn do_everything() { ... }
```

**RIGHT (Composition):**
```rust
fn install() { ... }
fn uninstall() { ... }
fn rebuild() {
    uninstall()?;  // REUSE
    install()?;    // REUSE
}
```

---

## RULE 6: RULE ZERO ≠ DELETE FEATURES

**RULE ZERO says:**
- Update existing functions, don't create `function_v2()`
- Break APIs if needed (pre-1.0)
- Delete deprecated code

**RULE ZERO does NOT say:**
- Delete all features
- Remove functionality
- Merge everything into one function

### Example

**WRONG:**
```rust
// Delete install_to_local_bin, create install_daemon
```

**RIGHT:**
```rust
// Update install_to_local_bin to be smarter
fn install_to_local_bin() {
    // Now builds if needed
}
```

---

## RULE 7: PRESERVE FUNCTIONALITY

**When refactoring:**

1. **List all current features** - What can users do now?
2. **Ensure all features still work** - After your changes
3. **Don't remove features** - Unless explicitly asked

### Checklist
- [ ] Can users still install?
- [ ] Can users still uninstall?
- [ ] Can users still rebuild?
- [ ] Does install work on fresh clone?
- [ ] Does rebuild actually install the binary?

---

## RULE 8: ONE CHANGE AT A TIME

**Don't do:**
- Delete 3 files
- Create 2 new files
- Refactor 5 functions
- Change the API

**All at once.**

**Do:**
1. Make one change
2. Verify it compiles
3. Verify it works
4. Then make the next change

---

## RULE 9: COMPILATION ERRORS ARE YOUR FRIEND

**When you break the API:**

1. Let the compiler find all call sites
2. Fix them one by one
3. Don't try to fix everything at once

**The compiler is smarter than you.**

---

## RULE 10: COST AWARENESS

**Every message costs money:**
- Your message: $0.05
- User's message: $0.05
- Total per exchange: $0.10

**A 10-message mistake costs $1.00**  
**A 80-message mistake costs $4.00**

### How to Save Money
1. Read carefully (saves 10 messages)
2. Ask questions (saves 20 messages)
3. Understand first (saves 30 messages)
4. Think about UX (saves 10 messages)

**Total savings:** $3.50 per task

---

## COMMON MISTAKES TO AVOID

### Mistake 1: "Let me delete this and start fresh"
**Why it's wrong:** You don't understand the existing code  
**What to do:** Read and understand first

### Mistake 2: "I'll create a new function for this"
**Why it's wrong:** Existing function might just need a fix  
**What to do:** Check existing functions first

### Mistake 3: "The user wants everything in one function"
**Why it's wrong:** They want code reuse, not deletion  
**What to do:** Use composition

### Mistake 4: "I'll fix this later"
**Why it's wrong:** You won't, and it'll break  
**What to do:** Fix it now

### Mistake 5: "This is obvious, I don't need to ask"
**Why it's wrong:** It's not obvious, you're guessing  
**What to do:** Ask the question

---

## CHECKLIST BEFORE EVERY CHANGE

- [ ] I read the user's message word-by-word
- [ ] I understand what they're asking for
- [ ] I read all the existing code
- [ ] I understand what the existing code does
- [ ] I identified the actual problem
- [ ] I thought about user experience
- [ ] I know my solution preserves all features
- [ ] I asked questions if anything was unclear
- [ ] I'm making ONE change, not ten

**If you can't check ALL boxes, STOP and ask questions.**

---

## EXAMPLES OF GOOD VS BAD

### Example 1: Code Duplication

**User:** "Can we consolidate build and install?"

**BAD Response:**
```
I'll delete build_daemon_local and install_to_local_bin 
and create one function that does both.
```

**GOOD Response:**
```
I see duplication in the build logic. Should I:
1. Make install_to_local_bin build automatically if needed?
2. Make build_daemon_local call install_to_local_bin?
3. Something else?
```

### Example 2: Unclear Request

**User:** "This is too complicated"

**BAD Response:**
```
I'll simplify by merging everything into one function.
```

**GOOD Response:**
```
What specifically is too complicated?
- Too many functions?
- Confusing names?
- Unclear what to call?
- Something else?
```

### Example 3: Feature Request

**User:** "Can rebuild also install?"

**BAD Response:**
```
I'll delete rebuild and create a new function that does both.
```

**GOOD Response:**
```
Should rebuild call install_to_local_bin after building?
That way it builds → installs → restarts.
```

---

## SUMMARY

1. **READ** - The actual words, not what you think they mean
2. **ASK** - Questions when unclear, don't guess
3. **UNDERSTAND** - Before touching code
4. **THINK** - About user experience
5. **COMPOSE** - Don't delete
6. **PRESERVE** - All features
7. **ONE CHANGE** - At a time
8. **SAVE MONEY** - Every message costs $0.05

---

## COST OF IGNORING THESE RULES

**TEAM-328 Example:**
- Task: 10 minutes
- Time taken: 30 minutes
- Messages: 80
- Cost: $4.00
- User frustration: MAXIMUM

**Following these rules:**
- Task: 10 minutes
- Time taken: 10 minutes
- Messages: 20
- Cost: $1.00
- User frustration: NONE

**Savings:** $3.00 + 20 minutes + user's sanity

---

**Read these rules. Follow these rules. Save everyone time and money.**
