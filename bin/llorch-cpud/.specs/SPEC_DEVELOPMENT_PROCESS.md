# Spec Development Process for llorch-cpud

**Date:** 2025-10-08  
**Status:** PROCESS DEFINITION

---

## Core Philosophy

**WE ARE NOT COPYING CODE LINE-BY-LINE FROM REFERENCES.**

This is the critical difference from llama-orchd development. We will:
- Study the references to understand the concepts
- Create specs that describe what needs to happen
- Let developers implement from the specs independently
- Validate each step against reference implementations

---

## Reference Implementations (Educational Focus)

We will use these references for **validation only**:

1. **candle** - Educational, clear structure
2. **mistral.rs** - Educational, good patterns
3. **tinygrad** - Educational, minimal complexity

**NOT llama.cpp** - Too advanced, production-grade, not educational

---

## The 5-Phase Development Process

### Phase 1: Study the References

**CRITICAL FIRST STEP:**
- **Use tinygrad's `examples/gpt2.py` as the reference** (simplest, most educational)
- **Trace the complete inference pipeline** for: GPT-2, FP32, CPU, temperature=0
- **Follow the ENTIRE code flow** from input token → output token
- **Document the complete flow in ONE markdown file** including:
  - Model initialization and weight loading
  - Token embedding lookup
  - Each transformer layer (attention + feedforward)
  - Final layer norm and LM head
  - Sampling/argmax (temp=0 means just argmax)
  - All tensor shapes at each step
  - All operations performed

**Why tinygrad/gpt2.py?**
- ~200 lines total - most readable
- Pure Python, minimal abstractions
- Clear data flow
- Educational focus (matches our philosophy)

**Then:**
- Cross-reference with candle and mistral.rs for validation
- Identify the critical validation points
- Document the architecture patterns

### Phase 2: Make the Specs from the Study
- Create detailed specifications for each component
- Define the order of implementation and testing
- Specify logging requirements for validation
- Document expected behaviors and outputs
- **NO CODE IN SPECS** - Only descriptions, requirements, and validation criteria

### Phase 3: Give the Specs to the Developers
- Hand off complete specifications
- Developers implement from specs independently
- Developers create their own code based on understanding the requirements

### Phase 4: Develop Each Spec One by One
- Implement one component at a time
- Complete implementation before moving to next spec
- Follow the defined order strictly
- Add comprehensive logging for validation

### Phase 5: Test Against the References Each Step
- Run reference implementations with special logging
- Compare outputs at each validation point
- Verify correctness before proceeding
- Document any deviations or learnings

---

## Special Logging Strategy

**In Reference Submodules:**
- Add detailed logging at each critical step
- Log tensor shapes, values, intermediate results
- Create validation checkpoints
- Make logging easy to compare with our implementation

**In llorch-cpud:**
- Mirror the logging structure
- Output comparable data at same checkpoints
- Enable step-by-step validation

---

## Team Structure

**Spec Team (Current Phase):**
- Study references
- Create specifications
- Define validation strategy
- Set implementation order

**Developer Team (Future Phase):**
- Receive specs
- Implement independently
- Create original code from requirements
- Validate against references

---

## Success Criteria

✅ Each component validated against references before moving forward  
✅ Clear understanding of why each step works  
✅ Original implementation, not copied code  
✅ Comprehensive logging for debugging  
✅ Step-by-step verification process  

---

## Current Status

**SPEC TEAM IS ACTIVE**

We are in Phase 1: Studying references and creating specifications.

**IMMEDIATE NEXT TASK:**
Someone needs to trace the complete code flow using **`reference/tinygrad/examples/gpt2.py`**:
- **Model:** GPT-2
- **Precision:** FP32
- **Device:** CPU
- **Temperature:** 0 (deterministic, simplest case)

**Reference file:** `/home/vince/Projects/llama-orch/reference/tinygrad/examples/gpt2.py` (~200 lines)

**Output:** ONE markdown file documenting the entire inference pipeline from input token to output token.

**DO NOT PROCEED TO IMPLEMENTATION YET.**

The developer team will receive complete specs before starting Phase 4.
