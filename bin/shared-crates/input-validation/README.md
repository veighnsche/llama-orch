# input-validation

**Centralized input validation and sanitization**

Validates all user inputs (model refs, paths, task IDs, prompts) to prevent injection attacks and path traversal.

**Key responsibilities:**
- Validate model_ref format (alphanumeric, length limits)
- Reject path traversal sequences (../, null bytes)
- Sanitize strings for logs (prevent log injection)
- Length limits on all string inputs
- Character whitelists per field type