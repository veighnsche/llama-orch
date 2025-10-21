#!/usr/bin/env python3
"""
TEAM-192: Automated migration script for job_router.rs
Converts Narration::new(ACTOR_QUEEN_ROUTER, ...) to NARRATE_ROUTER.narrate(...)
"""

import re
import sys

def migrate_narration(content):
    """Migrate Narration::new calls to NARRATE_ROUTER.narrate()"""
    
    # Pattern 1: Simple narration without format!()
    # Narration::new(ACTOR_QUEEN_ROUTER, "action", "target")
    #     .human("message")
    # → NARRATE_ROUTER.narrate("action")
    #     .human("message")
    
    pattern1 = r'Narration::new\(ACTOR_QUEEN_ROUTER,\s*"([^"]+)",\s*"[^"]+"\)\s*\n\s*\.human\("([^"]+)"\)'
    replacement1 = r'NARRATE_ROUTER.narrate("\1")\n                .human("\2")'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: Narration with &variable but no format!()
    # Narration::new(ACTOR_QUEEN_ROUTER, "action", &variable)
    #     .human("message")
    # → NARRATE_ROUTER.narrate("action")
    #     .human("message")
    
    pattern2 = r'Narration::new\(ACTOR_QUEEN_ROUTER,\s*"([^"]+)",\s*&[a-z_]+\)\s*\n\s*\.human\("([^"]+)"\)'
    replacement2 = r'NARRATE_ROUTER.narrate("\1")\n                .human("\2")'
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: Multi-line human message (no format)
    # Narration::new(ACTOR_QUEEN_ROUTER, "action", "target")
    #     .human(
    #         "multi\n\
    #          line"
    #     )
    # → NARRATE_ROUTER.narrate("action")
    #     .human(
    #         "multi\n\
    #          line"
    #     )
    
    pattern3 = r'Narration::new\(ACTOR_QUEEN_ROUTER,\s*"([^"]+)",\s*"[^"]+"\)\s*\n\s*\.human\(\s*\n'
    replacement3 = r'NARRATE_ROUTER.narrate("\1")\n                .human(\n'
    content = re.sub(pattern3, replacement3, content)
    
    return content

def main():
    file_path = "bin/10_queen_rbee/src/job_router.rs"
    
    print("TEAM-192: Migrating job_router.rs...")
    print(f"Reading {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Backup
    backup_path = file_path + ".backup"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup saved to {backup_path}")
    
    # Count original narrations
    original_count = content.count('Narration::new(ACTOR_QUEEN_ROUTER')
    print(f"Found {original_count} Narration::new calls")
    
    # Migrate
    migrated = migrate_narration(content)
    
    # Count remaining
    remaining_count = migrated.count('Narration::new(ACTOR_QUEEN_ROUTER')
    migrated_count = original_count - remaining_count
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(migrated)
    
    print(f"✅ Migrated {migrated_count}/{original_count} narrations")
    print(f"⚠️  {remaining_count} narrations still need manual migration (format!() calls)")
    print()
    print("Next steps:")
    print("1. Run: cargo check --bin queen-rbee")
    print("2. Manually fix remaining format!() calls")
    print("3. Add .context() calls for values")
    print("4. Replace format!() with {} placeholders")

if __name__ == "__main__":
    main()
