#!/usr/bin/env python3
"""Measure end-to-end CPU runtime and peak RSS for PARC scalability points.

The main benchmark records algorithm runtime. This script is a conservative
artifact sanity check that executes one deterministic PARC trial per requested
size in a fresh Python process and captures wall-clock time and max resident
set size via /usr/bin/time -v when available.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r"""
from scripts.run_benchmark import run_once
n=int(__import__('os').environ['PARC_MEM_N'])
seed=int(__import__('os').environ['PARC_MEM_SEED'])
rows,_=run_once(n,12,0.20,seed,methods=['parc'])
r=rows[0]
print(f"algorithm_runtime_sec={r['runtime_sec']:.6f}")
print(f"n_records={int(r['n_records'])}")
print(f"integrated_quality={r['integrated_quality']:.6f}")
print(f"aggregate_distortion={r['aggregate_distortion']:.6f}")
"""


def run_point(n: int, seed: int) -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['PARC_MEM_N'] = str(n)
    env['PARC_MEM_SEED'] = str(seed)
    cmd = ['/usr/bin/time', '-v', sys.executable, '-c', CHILD]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, check=True)
    stdout = proc.stdout
    stderr = proc.stderr
    def grab(name: str, text: str = stdout, default: str = 'nan') -> str:
        m = re.search(rf'^{re.escape(name)}=(.+)$', text, re.M)
        return m.group(1).strip() if m else default
    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', stderr)
    rss_kb = int(m.group(1)) if m else -1
    m = re.search(r'Elapsed \(wall clock\) time .*:\s*([0-9:.]+)', stderr)
    elapsed = m.group(1) if m else ''
    return {
        'n_entities': n,
        'seed': seed,
        'n_sources': 12,
        'budget': 0.20,
        'n_records': int(float(grab('n_records'))),
        'algorithm_runtime_sec': float(grab('algorithm_runtime_sec')),
        'integrated_quality': float(grab('integrated_quality')),
        'aggregate_distortion': float(grab('aggregate_distortion')),
        'peak_rss_mb': rss_kb / 1024.0 if rss_kb >= 0 else -1,
        'wall_clock_time': elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entities', type=int, nargs='+', default=[150, 900, 3000])
    ap.add_argument('--seed-base', type=int, default=52000)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'results' / 'runtime_memory_profile.csv'))
    args = ap.parse_args()
    rows = []
    for i, n in enumerate(args.entities):
        rows.append(run_point(n, args.seed_base + i))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
