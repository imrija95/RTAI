"""Public documentation integrity checks."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]*]\(([^)]+)\)")


def test_local_markdown_links_exist() -> None:
    missing = []
    for document in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(ROOT).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "missing local Markdown links:\n" + "\n".join(missing)


def test_repository_text_is_english_only() -> None:
    czech = re.compile(
        r"[\u00e1\u010d\u010f\u00e9\u011b\u00ed\u0148\u00f3\u0159\u0161\u0165"
        r"\u00fa\u016f\u00fd\u017e\u00c1\u010c\u010e\u00c9\u011a\u00cd\u0147"
        r"\u00d3\u0158\u0160\u0164\u00da\u016e\u00dd\u017d]"
    )
    offenders = []
    for pattern in ("*.py", "*.md"):
        for path in ROOT.rglob(pattern):
            if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
                continue
            if czech.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "non-English repository text: " + ", ".join(offenders)
