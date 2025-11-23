# Repository Timeline Generation for Investors

This tool generates a comprehensive timeline of repository evolution in 4-hour periods, perfect for investor presentations showing development progress without overwhelming detail.

## Quick Start

```bash
# Generate timeline (output to .timeline/ directory, default 4-hour periods)
./scripts/generate-timeline.sh

# Or use Python script directly with custom settings
python3 scripts/generate-hourly-timeline.py --output-dir investor-timeline --hours-per-period 4
```

## What You Get

### 1. Master Index (`INDEX.md`)
- Overall statistics (total commits, lines changed, time period)
- Period-by-period timeline table with links to detailed documents
- Daily summary showing activity patterns

### 2. Period Documents (`YYYY-MM-DD_HH00-HH00.md`)
- One document for each 4-hour period with commits
- Summary statistics (commits, files, lines changed)
- Detailed commit information with file changes
- Easy to navigate chronologically

## Output Structure

```
.timeline/
├── INDEX.md                      # Start here - Master timeline
├── 2024-01-15_0800-1200.md      # Morning period (08:00-12:00)
├── 2024-01-15_1200-1600.md      # Afternoon period (12:00-16:00)
├── 2024-01-15_1600-2000.md      # Evening period (16:00-20:00)
└── ...                           # One file per 4-hour period with activity
```

## Example Output

### INDEX.md Shows:

```markdown
# Repository Evolution Timeline

## Overall Statistics

- **Time Period**: 2024-01-15 08:00 to 2025-11-03 00:00
- **Total 4-Hour Periods**: 145
- **Total Commits**: 1,424
- **Average Commits/Period**: 9.8
- **Total Lines Added**: +2,695,767
- **Total Lines Removed**: -1,649,741
- **Net Change**: +1,046,026 lines

## 4-Hour Period Timeline

| Date       | Time Period | Commits | Files | +Lines | -Lines | Document |
|------------|-------------|---------|-------|--------|--------|----------|
| 2024-01-15 | 08:00-12:00 | 15      | 87    | +3420  | -456   | [View]   |
| 2024-01-15 | 12:00-16:00 | 23      | 142   | +5678  | -892   | [View]   |
...
```

### Period Document Shows:

```markdown
# 2024-01-15 08:00 - 12:00 Development Activity

## Summary

- **Commits**: 15
- **Files Modified**: 87
- **Lines Added**: +3420
- **Lines Removed**: -456
- **Net Change**: +2964 lines

## Commits

### 1. Initial project structure

**Time**: 09:15:23
**Author**: Vince
**Commit**: `a1b2c3d4`

Set up basic Rust workspace with cargo configuration.

**Files changed**:
- `Cargo.toml` (+45/-0)
- `src/main.rs` (+67/-0)
...
```

## Use Cases

### For Investor Meetings

1. **Show Timeline Evolution**: Open `INDEX.md` to show overall progress
2. **Dive into Details**: Click on specific hours to show detailed work
3. **Prove Consistent Development**: Daily summary shows regular activity
4. **Demonstrate Scale**: Statistics show lines of code, files touched

### For Progress Reports

- Use daily summary table for weekly/monthly reports
- Extract statistics for presentations
- Show development velocity trends

### For Due Diligence

- Complete audit trail of all development work
- Shows who did what and when
- Demonstrates code quality through commit messages
- Proves active development

## Advanced Usage

### Custom Output Directory

```bash
# For specific presentation
python3 scripts/generate-hourly-timeline.py --output-dir presentations/q4-2024

# For archival
python3 scripts/generate-hourly-timeline.py --output-dir docs/timeline-archive
```

### Integration with Git

```bash
# Generate timeline for specific branch
git checkout feature-branch
./scripts/generate-timeline.sh

# Generate timeline up to specific date
git log --before="2024-12-31" > /tmp/commits.log
# Then modify script to use this
```

### Code Files Only

The script shows only actual code files (Rust, TypeScript, JavaScript, Vue, etc.) and excludes:
- Configuration files (.json, .yaml, .toml)
- Documentation files (.md)
- Build/script files (.py, .sh, .lock)

This provides focused, code-only metrics for investor presentations.

## What Investors See

✅ **Consistent Activity**: 4-hour period grouping shows development patterns without overwhelming detail
✅ **Scale of Work**: Lines of code and file counts demonstrate significant effort
✅ **Quality**: Commit messages show thoughtful, structured development
✅ **Team Activity**: Multiple authors demonstrate team collaboration
✅ **Transparency**: Complete audit trail with comprehensive coverage
✅ **Digestible Format**: ~150 documents instead of 500+ (4x more manageable)

## Tips for Presentation

1. **Start with INDEX.md**: Shows big picture first
2. **Pick 3-5 Key Periods**: Choose 4-hour blocks with major milestones or heavy activity
3. **Use Daily Summary**: Shows consistent work patterns and velocity
4. **Highlight Net Lines**: Demonstrates codebase growth trajectory
5. **Print Statistics**: Put overall stats in slide deck for impact

## Maintenance

Re-generate timeline before investor meetings:

```bash
# Clean old timeline
rm -rf .timeline

# Generate fresh timeline
./scripts/generate-timeline.sh

# Review INDEX.md for latest statistics
cat .timeline/INDEX.md | head -30
```

## Technical Details

- **Language**: Python 3
- **Dependencies**: Only standard library (subprocess, datetime, pathlib)
- **Performance**: Processes ~1,000 commits in ~5 seconds
- **Output Format**: Markdown (GitHub/GitLab compatible)
- **Git Integration**: Uses `git log --all` to include all branches

## Troubleshooting

### Script fails to run

```bash
# Check Python version (need 3.6+)
python3 --version

# Make script executable
chmod +x scripts/generate-timeline.sh
chmod +x scripts/generate-hourly-timeline.py
```

### No output generated

```bash
# Check if you're in git repository
git status

# Check if there are commits
git log --oneline | head -5
```

### Output directory permission error

```bash
# Create directory manually
mkdir .timeline
chmod 755 .timeline

# Then run script
./scripts/generate-timeline.sh
```

## Customization

Edit `scripts/generate-hourly-timeline.py` to customize:

- **Line 89**: Change file limit (default: 20 files per commit)
- **Line 157**: Modify summary statistics shown
- **Line 200**: Adjust table columns in INDEX.md
- **Line 240**: Change daily summary format

---

**Generated by**: Repository Timeline Generator
**Purpose**: Investor presentation and progress tracking
**Maintained by**: Development team
