# TEAM-131 PEER REVIEW OF TEAM-132: Claim & Question Inventory

**Reviewing Team:** TEAM-131  
**Reviewed Team:** TEAM-132  
**Binary:** queen-rbee  
**Phase:** Day 1 Morning - Claim Extraction  
**Date:** 2025-10-19

---

## DOCUMENTS REVIEWED

1. ✅ `TEAM_132_INVESTIGATION_COMPLETE.md` (391 lines)
2. ✅ `TEAM_132_queen-rbee_INVESTIGATION_REPORT.md` (583 lines)
3. ✅ `TEAM_132_RISK_ANALYSIS.md` (742 lines)
4. ⏳ `TEAM_132_MIGRATION_PLAN.md` (not yet reviewed)

**Total Reviewed:** 1,716 lines across 3 documents

---

## ARCHITECTURE CLAIMS

### LOC Claims

1. **"Actual LOC: 2,015 lines of Rust code (17 files)"**
   - Source: INVESTIGATION_REPORT.md line 14
   - Status: ⏳ TO VERIFY with cloc

2. **"Total LOC (with comments/blanks): 2,719 lines"**
   - Source: INVESTIGATION_REPORT.md line 15
   - Status: ⏳ TO VERIFY

3. **"main.rs (283 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 45
   - Status: ⏳ TO VERIFY

4. **"beehive_registry.rs (200 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 47
   - Status: ⏳ TO VERIFY

5. **"worker_registry.rs (153 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 48
   - Status: ⏳ TO VERIFY

6. **"ssh.rs (76 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 49
   - Status: ⏳ TO VERIFY

7. **"http/inference.rs (466 LOC) - LARGEST"**
   - Source: INVESTIGATION_REPORT.md line 55
   - Status: ⏳ TO VERIFY

8. **"http/workers.rs (156 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 56
   - Status: ⏳ TO VERIFY

9. **"http/beehives.rs (146 LOC)"**
   - Source: INVESTIGATION_REPORT.md line 57
   - Status: ⏳ TO VERIFY

10. **"http/middleware/auth.rs (170 LOC)"**
    - Source: INVESTIGATION_REPORT.md line 60
    - Status: ⏳ TO VERIFY

11. **"preflight/rbee_hive.rs (76 LOC)"**
    - Source: INVESTIGATION_REPORT.md line 63
    - Status: ⏳ TO VERIFY

12. **"preflight/ssh.rs (60 LOC) - stub"**
    - Source: INVESTIGATION_REPORT.md line 64
    - Status: ⏳ TO VERIFY

### Crate LOC Claims

13. **"queen-rbee-registry: 353 LOC"**
    - Source: INVESTIGATION_COMPLETE.md line 30
    - Calculation: 200 + 153 = 353
    - Status: ⏳ TO VERIFY

14. **"queen-rbee-remote: 182 LOC"**
    - Source: INVESTIGATION_COMPLETE.md line 31
    - Calculation: 76 + 76 + 60 + 2 = ? (doesn't add up to 182)
    - Status: ⚠️ MATH ERROR? TO VERIFY

15. **"queen-rbee-http-server: 897 LOC"**
    - Source: INVESTIGATION_COMPLETE.md line 32
    - Status: ⏳ TO VERIFY - claimed to include "integration glue"

16. **"queen-rbee-orchestrator: 610 LOC"**
    - Source: INVESTIGATION_COMPLETE.md line 33
    - Status: ⏳ TO VERIFY

17. **"Binary cleanup: 283 LOC"**
    - Source: INVESTIGATION_COMPLETE.md line 34
    - Status: ⏳ TO VERIFY

### Dependency Claims

18. **"No circular dependencies detected"**
    - Source: INVESTIGATION_REPORT.md line 82
    - Status: ⏳ TO VERIFY with actual code analysis

19. **"Binary depends on all 4 crates"**
    - Source: INVESTIGATION_COMPLETE.md line 119-135
    - Status: ⏳ TO VERIFY

20. **"http-server depends only on registry"**
    - Source: RISK_ANALYSIS.md line 252
    - Status: ⏳ TO VERIFY

21. **"orchestrator depends on registry + remote"**
    - Source: RISK_ANALYSIS.md line 253
    - Status: ⏳ TO VERIFY

22. **"registry has no dependencies on other queen-rbee crates"**
    - Source: RISK_ANALYSIS.md line 250
    - Status: ⏳ TO VERIFY

---

## SHARED CRATE CLAIMS

### Used Shared Crates

23. **"auth-min: ✅ Used - Excellent - Full implementation"**
    - Source: INVESTIGATION_COMPLETE.md line 186
    - Status: ⏳ TO VERIFY with grep

24. **"input-validation: ✅ Used - Good - Validates requests"**
    - Source: INVESTIGATION_COMPLETE.md line 187
    - Status: ⏳ TO VERIFY

25. **"audit-logging: ✅ Used - Excellent - Auth events"**
    - Source: INVESTIGATION_COMPLETE.md line 188
    - Status: ⏳ TO VERIFY

26. **"deadline-propagation: ✅ Used - Excellent - Timeouts"**
    - Source: INVESTIGATION_COMPLETE.md line 189
    - Status: ⏳ TO VERIFY

27. **"secrets-management: ⚠️ Partial - Needs file-based token loading"**
    - Source: INVESTIGATION_COMPLETE.md line 190
    - Status: ⏳ TO VERIFY usage level

### Not Used Shared Crates

28. **"hive-core: ❌ Not used - Should share BeehiveNode type"**
    - Source: INVESTIGATION_COMPLETE.md line 191
    - Status: ⏳ TO VERIFY absence

29. **"model-catalog: ❌ Not used - Should query for model info"**
    - Source: INVESTIGATION_COMPLETE.md line 192
    - Status: ⏳ TO VERIFY absence

30. **"narration-core: ❌ Not used - Recommended for observability"**
    - Source: INVESTIGATION_COMPLETE.md line 193
    - Status: ⏳ TO VERIFY absence

31. **"5/10 security crates already integrated"**
    - Source: INVESTIGATION_COMPLETE.md line 22
    - Status: ⚠️ AMBIGUOUS - Which 10 crates? Need full list

### Shared Crate Details

32. **"auth-min used in http/middleware/auth.rs"**
    - Source: INVESTIGATION_REPORT.md line 436
    - Status: ⏳ TO VERIFY file location and usage

33. **"input-validation used in http/beehives.rs, http/inference.rs"**
    - Source: INVESTIGATION_REPORT.md line 438
    - Status: ⏳ TO VERIFY locations

34. **"audit-logging used in main.rs, http/middleware/auth.rs"**
    - Source: INVESTIGATION_REPORT.md line 439
    - Status: ⏳ TO VERIFY locations

35. **"deadline-propagation used in http/inference.rs"**
    - Source: INVESTIGATION_REPORT.md line 440
    - Status: ⏳ TO VERIFY

---

## TEST COVERAGE CLAIMS

36. **"Good test coverage - 11 tests across 8 modules"**
    - Source: INVESTIGATION_REPORT.md line 32
    - Status: ⏳ TO VERIFY count

37. **"beehive_registry::tests - CRUD operations (1 test)"**
    - Source: RISK_ANALYSIS.md line 293
    - Status: ⏳ TO VERIFY

38. **"worker_registry::tests - CRUD operations (1 test)"**
    - Source: RISK_ANALYSIS.md line 294
    - Status: ⏳ TO VERIFY

39. **"http/middleware/auth::tests - Authentication (4 tests)"**
    - Source: RISK_ANALYSIS.md line 295
    - Status: ⏳ TO VERIFY

40. **"http/routes::tests - Router creation (1 test)"**
    - Source: RISK_ANALYSIS.md line 297
    - Status: ⏳ TO VERIFY

41. **"http/health::tests - Health endpoint (1 test)"**
    - Source: RISK_ANALYSIS.md line 298
    - Status: ⏳ TO VERIFY

42. **"ssh::tests - Connection test (1 test, ignored)"**
    - Source: RISK_ANALYSIS.md line 299
    - Status: ⏳ TO VERIFY

43. **"preflight::tests - Preflight checks (2 tests)"**
    - Source: RISK_ANALYSIS.md line 300
    - Status: ⏳ TO VERIFY count

---

## SECURITY CLAIMS

44. **"Command injection vulnerability in ssh.rs:79"**
    - Source: INVESTIGATION_COMPLETE.md line 143
    - Status: ⏳ TO VERIFY exact location

45. **"TEAM-109 Audit Finding: Command injection via unsanitized user input"**
    - Source: INVESTIGATION_COMPLETE.md line 144
    - Status: ⏳ TO VERIFY audit document exists

46. **"Attack vector: Malicious admin adds node with crafted command"**
    - Source: RISK_ANALYSIS.md line 164
    - Status: ⏳ TO VERIFY exploit is possible

47. **"All endpoints protected by auth"**
    - Source: INVESTIGATION_COMPLETE.md line 177
    - Status: ⚠️ BROAD CLAIM - TO VERIFY all endpoints

---

## INTEGRATION CLAIMS

48. **"rbee-keeper → queen-rbee: HTTP, Stable, All endpoints protected"**
    - Source: INVESTIGATION_COMPLETE.md line 177
    - Status: ⏳ TO VERIFY integration exists

49. **"rbee-hive → queen-rbee: HTTP callbacks, Critical"**
    - Source: INVESTIGATION_COMPLETE.md line 178
    - Status: ⏳ TO VERIFY callbacks

50. **"queen-rbee → rbee-hive: HTTP client, Active, Worker spawning"**
    - Source: INVESTIGATION_COMPLETE.md line 179
    - Status: ⏳ TO VERIFY HTTP client usage

51. **"queen-rbee → workers: HTTP + SSE, Active, Inference execution"**
    - Source: INVESTIGATION_COMPLETE.md line 180
    - Status: ⏳ TO VERIFY SSE streaming

---

## PERFORMANCE CLAIMS

52. **"75-85% faster incremental builds (45-60s → 5-15s)"**
    - Source: INVESTIGATION_COMPLETE.md line 38
    - Status: ⚠️ PROJECTION - Cannot verify without migration

53. **"85-95% faster test iteration (45-60s → 2-8s)"**
    - Source: INVESTIGATION_COMPLETE.md line 39
    - Status: ⚠️ PROJECTION - Cannot verify

54. **"Estimated Full Rebuild: ~45-60 seconds"**
    - Source: INVESTIGATION_REPORT.md line 522
    - Status: ⏳ TO VERIFY with actual build

55. **"Registry build time: ~8s"**
    - Source: INVESTIGATION_REPORT.md line 528
    - Status: ⚠️ ESTIMATE - Cannot verify

56. **"Orchestrator build time: ~10s"**
    - Source: INVESTIGATION_REPORT.md line 531
    - Status: ⚠️ ESTIMATE - Cannot verify

---

## TIMELINE CLAIMS

57. **"4 crates, 2.5 days, 20 hours total"**
    - Source: INVESTIGATION_COMPLETE.md line 26
    - Status: ⚠️ ESTIMATE - Reasonable?

58. **"Phase 1 (Registry): 2h"**
    - Source: INVESTIGATION_COMPLETE.md line 248
    - Status: ⚠️ ESTIMATE

59. **"Phase 2 (Remote): 3h"**
    - Source: INVESTIGATION_COMPLETE.md line 249
    - Status: ⚠️ ESTIMATE

60. **"Phase 3 (HTTP): 4h"**
    - Source: INVESTIGATION_COMPLETE.md line 250
    - Status: ⚠️ ESTIMATE

61. **"Phase 4 (Orchestrator): 5h"**
    - Source: INVESTIGATION_COMPLETE.md line 251
    - Status: ⚠️ ESTIMATE

62. **"Phase 5 (Binary Cleanup): 1h"**
    - Source: INVESTIGATION_COMPLETE.md line 252
    - Status: ⚠️ ESTIMATE

63. **"33% buffer time (5 hours)"**
    - Source: INVESTIGATION_COMPLETE.md line 256
    - Status: ✅ REASONABLE

---

## ENDPOINT CLAIMS

64. **"GET /health"**
    - Source: RISK_ANALYSIS.md line 42
    - Status: ⏳ TO VERIFY exists

65. **"POST /v2/registry/beehives/add"**
    - Source: RISK_ANALYSIS.md line 43
    - Status: ⏳ TO VERIFY

66. **"GET /v2/registry/beehives/list"**
    - Source: RISK_ANALYSIS.md line 44
    - Status: ⏳ TO VERIFY

67. **"POST /v2/registry/beehives/remove"**
    - Source: RISK_ANALYSIS.md line 45
    - Status: ⏳ TO VERIFY

68. **"GET /v2/workers/list"**
    - Source: RISK_ANALYSIS.md line 46
    - Status: ⏳ TO VERIFY

69. **"GET /v2/workers/health"**
    - Source: RISK_ANALYSIS.md line 47
    - Status: ⏳ TO VERIFY

70. **"POST /v2/workers/shutdown"**
    - Source: RISK_ANALYSIS.md line 48
    - Status: ⏳ TO VERIFY

71. **"POST /v2/workers/register"**
    - Source: RISK_ANALYSIS.md line 49
    - Status: ⏳ TO VERIFY

72. **"POST /v2/workers/ready"**
    - Source: RISK_ANALYSIS.md line 50
    - Status: ⏳ TO VERIFY

73. **"POST /v2/tasks"**
    - Source: RISK_ANALYSIS.md line 51
    - Status: ⏳ TO VERIFY

74. **"POST /v1/inference"**
    - Source: RISK_ANALYSIS.md line 52
    - Status: ⏳ TO VERIFY

---

## FEATURE CLAIMS

75. **"TEAM-085: Localhost mode (no SSH required)"**
    - Source: INVESTIGATION_REPORT.md line 324
    - Status: ⏳ TO VERIFY feature exists

76. **"TEAM-087: Model reference validation (hf: prefix handling)"**
    - Source: INVESTIGATION_REPORT.md line 325
    - Status: ⏳ TO VERIFY

77. **"TEAM-093: Job ID injection for worker tracking"**
    - Source: INVESTIGATION_REPORT.md line 326
    - Status: ⏳ TO VERIFY

78. **"TEAM-124: Worker ready callback notifications (reduced timeout from 300s to 30s)"**
    - Source: INVESTIGATION_REPORT.md line 327
    - Status: ⏳ TO VERIFY timeout values

79. **"TEAM-114: Deadline propagation via x-deadline header"**
    - Source: INVESTIGATION_REPORT.md line 328
    - Status: ⏳ TO VERIFY header usage

---

## UNANSWERED QUESTIONS

### Questions to Other Teams

80. **"Can we share BeehiveNode type in hive-core?"**
    - Source: INVESTIGATION_COMPLETE.md line 305
    - Status: ❓ UNANSWERED - Need to investigate hive-core

81. **"Can we share WorkerSpawnRequest/Response types?"**
    - Source: INVESTIGATION_COMPLETE.md line 306
    - Status: ❓ UNANSWERED - Need to check rbee-hive

82. **"What's the best way to test rbee-hive callbacks?"**
    - Source: INVESTIGATION_COMPLETE.md line 307
    - Status: ❓ UNANSWERED - Need integration test approach

83. **"Should we extract ReadyResponse to shared crate?"**
    - Source: INVESTIGATION_COMPLETE.md line 310
    - Status: ❓ UNANSWERED - Need to check llm-worker-rbee

84. **"Do workers use any queen-rbee types directly?"**
    - Source: INVESTIGATION_COMPLETE.md line 311
    - Status: ❓ UNANSWERED - Need to check worker code

85. **"Does CLI import any queen-rbee code?"**
    - Source: INVESTIGATION_COMPLETE.md line 314
    - Status: ❓ UNANSWERED - Need to check rbee-keeper

86. **"Can we document API contract for CLI integration?"**
    - Source: INVESTIGATION_COMPLETE.md line 315
    - Status: ❓ UNANSWERED - Need API documentation

### Internal Questions

87. **"Is 20 hours realistic?"**
    - Source: INVESTIGATION_COMPLETE.md line 319
    - Status: ❓ OPINION QUESTION - Need peer validation

88. **"Should we merge registry + remote into one crate?"**
    - Source: INVESTIGATION_COMPLETE.md line 320
    - Status: ❓ DESIGN QUESTION - Need analysis

89. **"Do we need more integration tests?"**
    - Source: INVESTIGATION_COMPLETE.md line 321
    - Status: ❓ TEST COVERAGE QUESTION - Need gap analysis

90. **"Is command injection fix adequate?"**
    - Source: INVESTIGATION_COMPLETE.md line 322
    - Status: ❓ SECURITY QUESTION - Need code review

### Implicit Questions

91. **How does preflight/ssh.rs work if it's just a stub?**
    - Source: INVESTIGATION_REPORT.md line 419 "Mock implementation"
    - Status: ❓ NOT ADDRESSED - Need to investigate stub code

92. **What are ALL the shared crates available?**
    - Source: "5/10 security crates" but no complete list provided
    - Status: ❓ INCOMPLETE - Need full inventory

93. **Are there any other security vulnerabilities?**
    - Source: Only command injection mentioned
    - Status: ❓ INCOMPLETE AUDIT - Need full security scan

94. **What external dependencies does queen-rbee have?**
    - Source: Mentioned "20+ crates" but no list
    - Status: ❓ INCOMPLETE - Need Cargo.toml analysis

---

## TBD/TODO ITEMS

95. **"TODO: Use ModelInfo" in orchestrator.rs:127**
    - Source: INVESTIGATION_REPORT.md line 192 (implied from partial usage)
    - Status: 🚧 TBD - Need to verify this TODO exists

96. **"Preflight stub code - Future work: Implement real SSH2 library integration"**
    - Source: INVESTIGATION_REPORT.md line 421
    - Status: 🚧 FUTURE WORK - Documented but not planned

97. **"secrets-management needs file-based token loading"**
    - Source: INVESTIGATION_COMPLETE.md line 190
    - Status: 🚧 TBD - Not implemented yet

98. **"Should query catalog for model metadata"**
    - Source: INVESTIGATION_COMPLETE.md line 192
    - Status: 🚧 TBD - Recommended but not required

99. **"Recommended for observability" (narration-core)**
    - Source: INVESTIGATION_COMPLETE.md line 193
    - Status: 🚧 TBD - Optional enhancement

---

## ASSUMPTIONS WITHOUT PROOF

100. **"Clean architecture makes this low-risk"**
     - Source: INVESTIGATION_REPORT.md line 35
     - Status: ⚠️ ASSUMPTION - Need to verify architecture quality

101. **"Well-structured code with clear separation of concerns"**
     - Source: INVESTIGATION_REPORT.md line 27
     - Status: ⚠️ SUBJECTIVE - Need code review

102. **"No external API consumers"**
     - Source: RISK_ANALYSIS.md line 719
     - Status: ⚠️ ASSUMPTION - Need to verify rbee-keeper doesn't import

103. **"Probably handles timeouts correctly"**
     - Source: Not explicitly stated but implied by "deadline-propagation used"
     - Status: ⚠️ ASSUMPTION - Need to verify timeout logic

---

## MISSING INFORMATION

### Files Not Analyzed

104. **Migration Plan not reviewed yet**
     - File: `TEAM_132_MIGRATION_PLAN.md`
     - Status: ⏳ TO REVIEW

### Gaps in Analysis

105. **No dependency graph diagram provided**
     - Expected: Visual dependency graph
     - Actual: Text description only
     - Status: ⚠️ MISSING VISUALIZATION

106. **No actual build time measurements**
     - Expected: Benchmark results
     - Actual: Estimates only
     - Status: ⚠️ MISSING BASELINE DATA

107. **No binary size analysis**
     - Expected: Current binary size
     - Actual: Not mentioned
     - Status: ⚠️ MISSING DATA

108. **No memory usage analysis**
     - Expected: Runtime memory usage
     - Actual: Not mentioned
     - Status: ⚠️ MISSING DATA

109. **Incomplete shared crate list**
     - Expected: All 10+ shared crates checked
     - Actual: Only mentions 8 crates
     - Status: ⚠️ INCOMPLETE AUDIT

---

## CLAIM SUMMARY

**Total Claims Extracted:** 109

**Categories:**
- Architecture Claims: 22
- Shared Crate Claims: 13
- Test Coverage Claims: 8
- Security Claims: 4
- Integration Claims: 4
- Performance Claims: 5
- Timeline Claims: 9
- Endpoint Claims: 11
- Feature Claims: 5
- Unanswered Questions: 15
- TBD/TODO Items: 5
- Assumptions: 4
- Missing Information: 5

**Verification Status:**
- ⏳ TO VERIFY: 74 claims
- ⚠️ NEEDS INVESTIGATION: 20 claims
- ❓ UNANSWERED QUESTIONS: 15 items
- ✅ REASONABLE: 1 claim

---

## NEXT STEPS (Day 1 Afternoon)

1. **Verify LOC claims** - Run cloc on queen-rbee
2. **Verify file structure** - List all source files
3. **Verify dependencies** - Analyze Cargo.toml
4. **Verify shared crate usage** - grep for each shared crate
5. **Verify test count** - Find and count all tests
6. **Verify endpoints** - Search for route definitions
7. **Answer unanswered questions** - Investigate codebase

---

**Claim Inventory Status:** ✅ COMPLETE  
**Ready for:** Day 1 Afternoon Verification
