from __future__ import annotations

import json
from pathlib import Path

from experiments import config


def load_results(results_dir: Path | None = None) -> list[dict]:
    """Load result JSON files for analysis. [INV-4]"""
    root = results_dir or config.RESULTS_DIR
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Return deterministic min/max interval for small fixture samples."""
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


def ensure_paper_dirs() -> None:
    """Create paper output directories for generated artifacts."""
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    """Write a compact text-first SVG artifact."""
    height = 60 + 24 * len(lines)
    text_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{}">'.format(height),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="36" font-size="22" font-family="monospace">{title}</text>',
    ]
    for index, line in enumerate(lines):
        y = 72 + index * 24
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text_lines.append(f'<text x="24" y="{y}" font-size="15" font-family="monospace">{safe}</text>')
    text_lines.append("</svg>")
    path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
