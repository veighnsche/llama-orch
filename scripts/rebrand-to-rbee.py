#!/usr/bin/env python3
"""
Rebrand Script: rbees → rbee (with specialized binary names)

New naming scheme:
- rbees → rbee (main brand)
- rbees-workerd → llm-worker-rbee (specialized worker type)
- rbees-pool → rbee-hive (pool manager)
- rbees-ctl → rbee-keeper (CLI, binary: rbee)
- rbees-orcd → queen-rbee (orchestrator daemon)

This script performs automated find-and-replace for the rbee rebrand.
It's designed to be safe, reversible, and testable.

Usage:
    python3 scripts/rebrand-to-rbee.py --dry-run    # Preview changes
    python3 scripts/rebrand-to-rbee.py              # Execute changes
    python3 scripts/rebrand-to-rbee.py --verify     # Verify no old refs remain

Safety:
    - Run on git branch (rebrand/rbee-naming)
    - Commit after each phase for easy rollback
    - Dry-run mode shows changes without applying
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
from dataclasses import dataclass


@dataclass
class ReplaceRule:
    """A find-and-replace rule with context"""
    pattern: str
    replacement: str
    description: str
    file_patterns: List[str]
    exclude_patterns: List[str] = None
    is_regex: bool = False


class RebrandScript:
    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.changes_made = 0
        self.files_modified = set()
        
        # Patterns to exclude
        self.exclude_dirs = {'.git', 'target', '.business', 'node_modules', '.venv', 'reference'}
        self.exclude_files = {'Cargo.lock', 'pnpm-lock.yaml', 'rebrand-to-rbees.py', 'rebrand-to-rbee.py'}
        
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        # Skip excluded directories
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return False
        
        # Skip excluded files
        if file_path.name in self.exclude_files:
            return False
        
        return True
    
    def find_files(self, patterns: List[str], exclude_patterns: List[str] = None) -> List[Path]:
        """Find files matching patterns"""
        files = []
        exclude_patterns = exclude_patterns or []
        
        for pattern in patterns:
            if pattern.startswith('*.'):
                # Extension pattern
                ext = pattern[1:]  # Remove *
                for file_path in self.repo_root.rglob(f'*{ext}'):
                    # Skip directories
                    if file_path.is_dir():
                        continue
                    if self.should_process_file(file_path):
                        # Check exclude patterns
                        should_exclude = False
                        for exclude in exclude_patterns:
                            if exclude in str(file_path):
                                should_exclude = True
                                break
                        if not should_exclude:
                            files.append(file_path)
            else:
                # Specific file pattern
                for file_path in self.repo_root.rglob(pattern):
                    # Skip directories
                    if file_path.is_dir():
                        continue
                    if self.should_process_file(file_path):
                        files.append(file_path)
        
        return sorted(set(files))
    
    def replace_in_file(self, file_path: Path, pattern: str, replacement: str, is_regex: bool = False) -> int:
        """Replace pattern in file, return number of replacements"""
        # Skip directories
        if file_path.is_dir():
            return 0
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError, IsADirectoryError):
            # Skip binary files, files we can't read, or directories
            return 0
        
        if is_regex:
            new_content, count = re.subn(pattern, replacement, content)
        else:
            # Literal string replacement
            count = content.count(pattern)
            new_content = content.replace(pattern, replacement)
        
        if count > 0:
            if not self.dry_run:
                file_path.write_text(new_content, encoding='utf-8')
            self.files_modified.add(file_path)
            return count
        
        return 0
    
    def apply_rule(self, rule: ReplaceRule) -> Tuple[int, int]:
        """Apply a replacement rule, return (files_changed, total_replacements)"""
        print(f"\n{'[DRY RUN] ' if self.dry_run else ''}Applying: {rule.description}")
        
        files = self.find_files(rule.file_patterns, rule.exclude_patterns)
        print(f"  Found {len(files)} files to process")
        
        total_replacements = 0
        files_changed = 0
        
        for file_path in files:
            count = self.replace_in_file(file_path, rule.pattern, rule.replacement, rule.is_regex)
            if count > 0:
                total_replacements += count
                files_changed += 1
                rel_path = file_path.relative_to(self.repo_root)
                print(f"    {rel_path}: {count} replacement{'s' if count > 1 else ''}")
        
        print(f"  Total: {total_replacements} replacements in {files_changed} files")
        self.changes_made += total_replacements
        
        return files_changed, total_replacements


def get_phase1_rules() -> List[ReplaceRule]:
    """Phase 1: Critical - Rust source code"""
    return [
        # 1. Library name in imports
        ReplaceRule(
            pattern='rbees_workerd',
            replacement='llm_worker_rbee',
            description='Rust library name: rbees_workerd → llm_worker_rbee',
            file_patterns=['*.rs'],
        ),
        
        # 2. Binary names in strings
        ReplaceRule(
            pattern='rbees-workerd',
            replacement='llm-worker-rbee',
            description='Binary name in strings: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.rs'],
        ),
        
        ReplaceRule(
            pattern='rbees-pool',
            replacement='rbee-hive',
            description='Binary name in strings: rbees-pool → rbee-hive',
            file_patterns=['*.rs'],
        ),
        
        ReplaceRule(
            pattern='rbees-ctl',
            replacement='rbee-keeper',
            description='Binary name in strings: rbees-ctl → rbee-keeper',
            file_patterns=['*.rs'],
        ),
        
        ReplaceRule(
            pattern='rbees-orcd',
            replacement='queen-rbee',
            description='Binary name in strings: rbees-orcd → queen-rbee',
            file_patterns=['*.rs'],
        ),
        
        # 3. Actor constants
        ReplaceRule(
            pattern='ACTOR_RBEES_WORKERD',
            replacement='ACTOR_LLM_WORKER_RBEE',
            description='Actor constant: ACTOR_RBEES_WORKERD → ACTOR_LLM_WORKER_RBEE',
            file_patterns=['*.rs'],
        ),
        
        # 4. Clap command names
        ReplaceRule(
            pattern=r'#\[command\(name = "rbees-workerd"\)\]',
            replacement='#[command(name = "llm-worker-rbee")]',
            description='Clap command name: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.rs'],
            is_regex=True,
        ),
        
        ReplaceRule(
            pattern=r'#\[command\(name = "rbees"\)\]',
            replacement='#[command(name = "rbee")]',
            description='Clap command name: rbees → rbee',
            file_patterns=['*.rs'],
            is_regex=True,
        ),
        
        ReplaceRule(
            pattern=r'#\[command\(name = "rbees-pool"\)\]',
            replacement='#[command(name = "rbee-hive")]',
            description='Clap command name: rbees-pool → rbee-hive',
            file_patterns=['*.rs'],
            is_regex=True,
        ),
        
        ReplaceRule(
            pattern=r'#\[command\(name = "rbees-orcd"\)\]',
            replacement='#[command(name = "queen-rbee")]',
            description='Clap command name: rbees-orcd → queen-rbee',
            file_patterns=['*.rs'],
            is_regex=True,
        ),
        
        # 5. Brand name (standalone)
        ReplaceRule(
            pattern='rbees',
            replacement='rbee',
            description='Brand name: rbees → rbee (general)',
            file_patterns=['*.rs'],
        ),
    ]


def get_phase2_rules() -> List[ReplaceRule]:
    """Phase 2: Shell scripts and config files"""
    return [
        # Shell scripts
        ReplaceRule(
            pattern='rbees-workerd',
            replacement='llm-worker-rbee',
            description='Shell scripts: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.sh'],
        ),
        
        ReplaceRule(
            pattern='rbees-pool',
            replacement='rbee-hive',
            description='Shell scripts: rbees-pool → rbee-hive',
            file_patterns=['*.sh'],
        ),
        
        ReplaceRule(
            pattern='rbees-ctl',
            replacement='rbee-keeper',
            description='Shell scripts: rbees-ctl → rbee-keeper',
            file_patterns=['*.sh'],
        ),
        
        ReplaceRule(
            pattern='rbees-orcd',
            replacement='queen-rbee',
            description='Shell scripts: rbees-orcd → queen-rbee',
            file_patterns=['*.sh'],
        ),
        
        ReplaceRule(
            pattern='rbees',
            replacement='rbee',
            description='Shell scripts: rbees → rbee',
            file_patterns=['*.sh'],
        ),
        
        # Config files (non-Cargo.toml)
        ReplaceRule(
            pattern='rbees-workerd',
            replacement='llm-worker-rbee',
            description='Config files: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.toml'],
            exclude_patterns=['Cargo.toml'],
        ),
        
        ReplaceRule(
            pattern='rbees-pool',
            replacement='rbee-hive',
            description='Config files: rbees-pool → rbee-hive',
            file_patterns=['*.toml'],
            exclude_patterns=['Cargo.toml'],
        ),
        
        ReplaceRule(
            pattern='rbees',
            replacement='rbee',
            description='Config files: rbees → rbee',
            file_patterns=['*.toml'],
            exclude_patterns=['Cargo.toml'],
        ),
    ]


def get_phase3_rules() -> List[ReplaceRule]:
    """Phase 3: Documentation"""
    return [
        # Markdown files - specific binaries first
        ReplaceRule(
            pattern='rbees-workerd',
            replacement='llm-worker-rbee',
            description='Documentation: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.md'],
        ),
        
        ReplaceRule(
            pattern='rbees-pool',
            replacement='rbee-hive',
            description='Documentation: rbees-pool → rbee-hive',
            file_patterns=['*.md'],
        ),
        
        ReplaceRule(
            pattern='rbees-ctl',
            replacement='rbee-keeper',
            description='Documentation: rbees-ctl → rbee-keeper',
            file_patterns=['*.md'],
        ),
        
        ReplaceRule(
            pattern='rbees-orcd',
            replacement='queen-rbee',
            description='Documentation: rbees-orcd → queen-rbee',
            file_patterns=['*.md'],
        ),
        
        # CPU/CUDA/Metal worker variants
        ReplaceRule(
            pattern='rbees-cpu-workerd',
            replacement='llm-worker-rbee-cpu',
            description='Documentation: rbees-cpu-workerd → llm-worker-rbee-cpu',
            file_patterns=['*.md'],
        ),
        
        ReplaceRule(
            pattern='rbees-cuda-workerd',
            replacement='llm-worker-rbee-cuda',
            description='Documentation: rbees-cuda-workerd → llm-worker-rbee-cuda',
            file_patterns=['*.md'],
        ),
        
        ReplaceRule(
            pattern='rbees-metal-workerd',
            replacement='llm-worker-rbee-metal',
            description='Documentation: rbees-metal-workerd → llm-worker-rbee-metal',
            file_patterns=['*.md'],
        ),
        
        # General brand name
        ReplaceRule(
            pattern=r'`rbees`',
            replacement='`rbee`',
            description='Documentation: `rbees` → `rbee`',
            file_patterns=['*.md'],
            is_regex=True,
        ),
        
        ReplaceRule(
            pattern=r'\brbees\b',
            replacement='rbee',
            description='Documentation: rbees (word) → rbee',
            file_patterns=['*.md'],
            is_regex=True,
        ),
    ]


def get_phase4_rules() -> List[ReplaceRule]:
    """Phase 4: YAML/CI files"""
    return [
        ReplaceRule(
            pattern='rbees-workerd',
            replacement='llm-worker-rbee',
            description='CI/YAML: rbees-workerd → llm-worker-rbee',
            file_patterns=['*.yml', '*.yaml'],
        ),
        
        ReplaceRule(
            pattern='rbees-pool',
            replacement='rbee-hive',
            description='CI/YAML: rbees-pool → rbee-hive',
            file_patterns=['*.yml', '*.yaml'],
        ),
        
        ReplaceRule(
            pattern='rbees-ctl',
            replacement='rbee-keeper',
            description='CI/YAML: rbees-ctl → rbee-keeper',
            file_patterns=['*.yml', '*.yaml'],
        ),
        
        ReplaceRule(
            pattern='rbees-orcd',
            replacement='queen-rbee',
            description='CI/YAML: rbees-orcd → queen-rbee',
            file_patterns=['*.yml', '*.yaml'],
        ),
        
        ReplaceRule(
            pattern='rbees',
            replacement='rbee',
            description='CI/YAML: rbees → rbee',
            file_patterns=['*.yml', '*.yaml'],
        ),
    ]


def verify_no_old_references(repo_root: Path) -> bool:
    """Verify no old references remain"""
    print("\n" + "="*80)
    print("VERIFICATION: Checking for remaining old references")
    print("="*80)
    
    patterns_to_check = [
        'rbees_workerd',
        'rbees-workerd',
        'rbees-pool',
        'rbees-ctl',
        'rbees-orcd',
        'ACTOR_RBEES_WORKERD',
        r'\brbees\b',  # Word boundary to avoid matching "rbee"
    ]
    
    exclude_dirs = {'.git', 'target', '.business', 'node_modules', '.venv', 'reference'}
    exclude_files = {'rebrand-to-rbees.py', 'rebrand-to-rbee.py'}
    
    found_issues = False
    
    for pattern in patterns_to_check:
        print(f"\nChecking for: {pattern}")
        matches = []
        is_regex = pattern.startswith('\\b')
        
        for file_path in repo_root.rglob('*'):
            # Skip directories and excluded paths
            if file_path.is_dir():
                continue
            
            if file_path.name in exclude_files:
                continue
            
            skip = False
            for part in file_path.parts:
                if part in exclude_dirs:
                    skip = True
                    break
            if skip:
                continue
            
            # Skip binary files
            if file_path.suffix in {'.lock', '.png', '.jpg', '.gif', '.pdf'}:
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                if is_regex:
                    if re.search(pattern, content):
                        count = len(re.findall(pattern, content))
                        matches.append((file_path, count))
                else:
                    if pattern in content:
                        count = content.count(pattern)
                        matches.append((file_path, count))
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if matches:
            found_issues = True
            print(f"  ❌ Found {len(matches)} files with '{pattern}':")
            for file_path, count in matches[:10]:  # Show first 10
                rel_path = file_path.relative_to(repo_root)
                print(f"     {rel_path}: {count} occurrence{'s' if count > 1 else ''}")
            if len(matches) > 10:
                print(f"     ... and {len(matches) - 10} more files")
        else:
            print(f"  ✅ No occurrences found")
    
    print("\n" + "="*80)
    if found_issues:
        print("❌ VERIFICATION FAILED: Old references still exist")
        print("="*80)
        return False
    else:
        print("✅ VERIFICATION PASSED: No old references found")
        print("="*80)
        return True


def rename_directories(repo_root: Path, dry_run: bool = False) -> None:
    """Rename directories"""
    print("\n" + "="*80)
    print(f"{'[DRY RUN] ' if dry_run else ''}Renaming directories")
    print("="*80)
    
    renames = [
        ('bin/rbees-workerd', 'bin/llm-worker-rbee'),
        ('bin/rbees-pool', 'bin/rbee-hive'),
        ('bin/rbees-ctl', 'bin/rbee-keeper'),
        ('bin/rbees-orcd', 'bin/queen-rbee'),
    ]
    
    for old_path, new_path in renames:
        old_dir = repo_root / old_path
        new_dir = repo_root / new_path
        
        if old_dir.exists():
            print(f"  {old_path} → {new_path}")
            if not dry_run:
                old_dir.rename(new_dir)
        else:
            print(f"  ⚠️  {old_path} not found (may already be renamed)")


def rename_config_files(repo_root: Path, dry_run: bool = False) -> None:
    """Rename config files"""
    print("\n" + "="*80)
    print(f"{'[DRY RUN] ' if dry_run else ''}Renaming config files")
    print("="*80)
    
    renames = [
        ('bin/llm-worker-rbee/.rbees-test.toml', 'bin/llm-worker-rbee/.rbee-test.toml'),
    ]
    
    for old_path, new_path in renames:
        old_file = repo_root / old_path
        new_file = repo_root / new_path
        
        if old_file.exists():
            print(f"  {old_path} → {new_path}")
            if not dry_run:
                old_file.rename(new_file)
        else:
            print(f"  ⚠️  {old_path} not found (may already be renamed)")


def main():
    parser = argparse.ArgumentParser(description='Rebrand rbees to rbee with specialized names')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--verify', action='store_true', help='Verify no old references remain')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], help='Run specific phase only')
    args = parser.parse_args()
    
    # Find repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    print("="*80)
    print("rbee Rebrand Script (rbees → rbee with specialized names)")
    print("="*80)
    print(f"Repository: {repo_root}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'EXECUTE (will modify files)'}")
    print("="*80)
    print("\nNew naming scheme:")
    print("  rbees-workerd → llm-worker-rbee (LLM worker type)")
    print("  rbees-pool    → rbee-hive (pool manager)")
    print("  rbees-ctl     → rbee-keeper (CLI, binary: rbee)")
    print("  rbees-orcd    → queen-rbee (orchestrator daemon)")
    print("="*80)
    
    if args.verify:
        success = verify_no_old_references(repo_root)
        sys.exit(0 if success else 1)
    
    script = RebrandScript(repo_root, dry_run=args.dry_run)
    
    phases = {
        1: ("Phase 1: Critical - Rust source code", get_phase1_rules()),
        2: ("Phase 2: Shell scripts and config files", get_phase2_rules()),
        3: ("Phase 3: Documentation", get_phase3_rules()),
        4: ("Phase 4: CI/YAML files", get_phase4_rules()),
    }
    
    # Rename directories first (before code changes)
    if not args.phase or args.phase == 1:
        rename_directories(repo_root, dry_run=args.dry_run)
    
    # Run specific phase or all phases
    phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4]
    
    for phase_num in phases_to_run:
        title, rules = phases[phase_num]
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        for rule in rules:
            script.apply_rule(rule)
    
    # Rename config files
    if not args.phase or args.phase == 2:
        rename_config_files(repo_root, dry_run=args.dry_run)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total changes: {script.changes_made}")
    print(f"Files modified: {len(script.files_modified)}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE: No files were actually modified")
        print("Run without --dry-run to apply changes")
    else:
        print("\n✅ Changes applied successfully")
        print("\nNext steps:")
        print("  1. Update Cargo.toml workspace members")
        print("  2. Update Cargo.toml package names")
        print("  3. Test compilation: cargo build --workspace")
        print("  4. Run verification: python3 scripts/rebrand-to-rbee.py --verify")
        print("  5. Commit changes: git add -A && git commit -m 'Rebrand: rbees → rbee with specialized names'")
        print("  6. If issues found: git reset --hard HEAD~1")
    
    print("="*80)


if __name__ == '__main__':
    main()
