"""Dev-only repository scanner for context index (never used at runtime in 'run')."""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List

TEXT_EXTS = {'.md', '.mdx', '.txt', '.yaml', '.yml', '.json'}
SECRETS_RE = re.compile(r"(BEGIN (RSA|EC|OPENSSH) PRIVATE KEY|api[_-]?key|secret|token)", re.I)
DOMAIN_TERMS = [
    'omzet', 'cogs', 'marge', 'opex', 'huur', 'marketing', 'salarissen',
    'verzekeringen', 'ict', 'overig', 'dso', 'dpo', 'btw', 'kor',
    'urencriterium', 'ib', 'vpb', 'qredits', 'amortisatie', 'afschrijving',
    'liquiditeit', 'exploitatie', 'seizoen', 'krediet', 'lening', 'hoofdsom',
    'rente', 'grace', 'aflossing'
]


def _extract_snippets(txt: str, term: str, max_snips: int = 2, span: int = 200) -> List[str]:
    ltxt = txt.lower()
    res: List[str] = []
    start = 0
    while len(res) < max_snips:
        idx = ltxt.find(term, start)
        if idx == -1:
            break
        a = max(0, idx - span // 2)
        b = min(len(txt), idx + span // 2)
        snip = txt[a:b]
        snip = snip.replace('\n', ' ').replace('\r', ' ')
        snip = re.sub(r"\s+", " ", snip).strip()
        res.append(snip)
        start = idx + len(term)
    return res


def scan_roots(roots: List[Path], out_dir: Path) -> None:
    files_meta: List[Dict[str, Any]] = []
    term_counts: Dict[str, int] = {t: 0 for t in DOMAIN_TERMS}
    snippets: Dict[str, List[str]] = {t: [] for t in DOMAIN_TERMS}

    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in TEXT_EXTS:
                    continue
                try:
                    st = p.stat()
                except Exception:
                    continue
                if st.st_size > 2 * 1024 * 1024:  # 2MB
                    continue
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if SECRETS_RE.search(txt):
                    continue
                files_meta.append({
                    'path': str(p.resolve()),
                    'relpath': str(p),
                    'size': st.st_size,
                    'mtime': int(st.st_mtime),
                })
                ltxt = txt.lower()
                for t in DOMAIN_TERMS:
                    c = ltxt.count(t)
                    if c > 0:
                        term_counts[t] += c
                        if len(snippets[t]) < 2:
                            snippets[t].extend(_extract_snippets(txt, t))
                            snippets[t] = snippets[t][:2]

    top_terms = sorted(term_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_terms_dict = {k: v for k, v in top_terms if v > 0}
    top_snippets = {k: snippets[k] for k in top_terms_dict.keys()}

    out = {
        'files': files_meta,
        'terms': term_counts,
        'top_terms': top_terms_dict,
        'snippets': top_snippets,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'context_index.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
