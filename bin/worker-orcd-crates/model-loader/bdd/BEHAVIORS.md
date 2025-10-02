# model-loader Behaviors

**Observable behaviors tested via BDD**

## Hash Verification Behaviors

**B-HASH-001**: When loading a model with correct hash, the load succeeds  
**B-HASH-002**: When loading a model with wrong hash, the load fails with HashMismatch  
**B-HASH-003**: When loading a model without hash, the load succeeds (optional verification)

## GGUF Format Validation Behaviors

**B-GGUF-001**: When loading a valid GGUF file, the load succeeds  
**B-GGUF-002**: When loading a file with invalid magic number, the load fails with InvalidFormat  
**B-GGUF-003**: When validating valid GGUF bytes in memory, the validation succeeds  
**B-GGUF-004**: When validating invalid GGUF bytes in memory, the validation fails with InvalidFormat

## Resource Limit Behaviors

**B-LIMIT-001**: When loading a file exceeding max_size, the load fails with TooLarge  
**B-LIMIT-002**: (TODO M0) When loading a GGUF with excessive tensor count, the load fails  
**B-LIMIT-003**: (TODO M0) When loading a GGUF with oversized strings, the load fails

## Path Security Behaviors

**B-PATH-001**: (TODO M0) When loading a path with traversal sequence, the load fails with PathValidationFailed  
**B-PATH-002**: (TODO M0) When loading a symlink outside allowed directory, the load fails  
**B-PATH-003**: (TODO M0) When loading a path with null byte, the load fails

## Error Message Behaviors

**B-ERROR-001**: Error messages are actionable and specific  
**B-ERROR-002**: Error messages do not expose sensitive data (file contents, paths)  
**B-ERROR-003**: Error messages distinguish retriable from fatal errors
