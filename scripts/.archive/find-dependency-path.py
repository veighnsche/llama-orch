#!/usr/bin/env python3
"""
Find dependency paths between two packages in the monorepo.

Usage:
    python scripts/find-dependency-path.py <from-package> <to-package>
    
Example:
    python scripts/find-dependency-path.py rbee-keeper observability-narration-core
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import deque


def load_dependency_graph() -> Dict:
    """Load dependency graph from JSON file or generate it"""
    deps_file = Path("/tmp/deps.json")
    
    if not deps_file.exists():
        print("🔍 Generating dependency graph...")
        import subprocess
        script_dir = Path(__file__).parent
        subprocess.run([
            "python3",
            str(script_dir / "dependency-graph.py"),
            "--format", "json",
            "--output", str(deps_file)
        ], check=True, capture_output=True)
        
    with open(deps_file) as f:
        return json.load(f)


def find_all_paths(
    graph: Dict,
    start: str,
    end: str,
    max_depth: int = 10
) -> List[List[str]]:
    """Find all paths from start to end package using BFS"""
    
    if start not in graph['packages']:
        print(f"❌ Package '{start}' not found in workspace")
        return []
        
    if end not in graph['packages']:
        print(f"❌ Package '{end}' not found in workspace")
        return []
        
    # Build adjacency list
    adj: Dict[str, Set[str]] = {}
    for pkg_name, pkg_data in graph['packages'].items():
        adj[pkg_name] = set(pkg_data['dependencies'] + pkg_data['devDependencies'])
        
    # BFS to find all paths
    paths: List[List[str]] = []
    queue = deque([([start], set([start]))])
    
    while queue:
        path, visited = queue.popleft()
        current = path[-1]
        
        if len(path) > max_depth:
            continue
            
        if current == end:
            paths.append(path)
            continue
            
        for neighbor in adj.get(current, set()):
            if neighbor not in visited:
                new_visited = visited | {neighbor}
                queue.append((path + [neighbor], new_visited))
                
    return paths


def find_shortest_path(
    graph: Dict,
    start: str,
    end: str
) -> Optional[List[str]]:
    """Find shortest path using BFS"""
    paths = find_all_paths(graph, start, end)
    if not paths:
        return None
    return min(paths, key=len)


def format_path(path: List[str], graph: Dict) -> str:
    """Format a dependency path for display"""
    lines = []
    for i, pkg in enumerate(path):
        pkg_type = graph['packages'][pkg]['type']
        pkg_path = graph['packages'][pkg]['path']
        
        indent = "  " * i
        arrow = "└─> " if i > 0 else ""
        
        lines.append(f"{indent}{arrow}{pkg} ({pkg_type})")
        lines.append(f"{indent}    📁 {pkg_path}")
        
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/find-dependency-path.py <from-package> <to-package>")
        print()
        print("Example:")
        print("  python scripts/find-dependency-path.py rbee-keeper observability-narration-core")
        sys.exit(1)
        
    start_pkg = sys.argv[1]
    end_pkg = sys.argv[2]
    
    print(f"🔍 Finding dependency paths from '{start_pkg}' to '{end_pkg}'...")
    print()
    
    graph = load_dependency_graph()
    
    # Find shortest path
    shortest = find_shortest_path(graph, start_pkg, end_pkg)
    
    if not shortest:
        print(f"❌ No dependency path found from '{start_pkg}' to '{end_pkg}'")
        print()
        print("This could mean:")
        print(f"  - '{start_pkg}' does not depend on '{end_pkg}' (directly or transitively)")
        print(f"  - One or both packages don't exist in the workspace")
        sys.exit(1)
        
    # Find all paths (up to 10 hops)
    all_paths = find_all_paths(graph, start_pkg, end_pkg, max_depth=10)
    
    print(f"✅ Found {len(all_paths)} dependency path(s)")
    print()
    
    # Show shortest path
    print("## Shortest Path")
    print(f"**Length:** {len(shortest) - 1} hop(s)")
    print()
    print(format_path(shortest, graph))
    print()
    
    # Show alternative paths if they exist
    if len(all_paths) > 1:
        print("## Alternative Paths")
        print()
        
        for i, path in enumerate(sorted(all_paths, key=len)[1:6], 1):  # Show up to 5 more
            print(f"### Path {i + 1} ({len(path) - 1} hops)")
            print()
            print(format_path(path, graph))
            print()
            
        if len(all_paths) > 6:
            print(f"... and {len(all_paths) - 6} more path(s)")
            print()
            
    # Show reverse dependencies
    print("## Reverse Check")
    reverse_path = find_shortest_path(graph, end_pkg, start_pkg)
    if reverse_path:
        print(f"⚠️  **Circular dependency detected!**")
        print(f"'{end_pkg}' also depends on '{start_pkg}' (directly or transitively)")
        print()
        print(format_path(reverse_path, graph))
    else:
        print(f"✅ No circular dependency ('{end_pkg}' does not depend on '{start_pkg}')")
        
    print()


if __name__ == '__main__':
    main()
