#!/usr/bin/env python3
"""Regenerate the paper architecture figure.

Kept as a convenience entry point; the canonical plotting implementation lives in
scripts/plot_results.py so the diagram and experiment figures share typography.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_results import plot_corruption_model  # noqa: E402

if __name__ == "__main__":
    plot_corruption_model(ROOT / "figures")
    print(ROOT / "figures" / "corruption_model.pdf")
