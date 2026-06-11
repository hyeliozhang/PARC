#!/usr/bin/env python3
"""Create statistical evidence tables for the PARC paper.

The script consumes benchmark CSV files already produced by run_benchmark.py or
run_full_benchmark.py. It does not rerun experiments and does not edit results;
it only computes paired improvements, bootstrap confidence intervals, and a
compact set of paper-ready tables.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

METRICS = ["integrated_quality", "cell_accuracy", "aggregate_distortion", "false_positive_rate", "runtime_sec"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_test_greater(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0
    k = int((diffs > 0).sum())
    # one-sided exact binomial P[X >= k], p=0.5
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def paired_tests(df: pd.DataFrame, baseline: str, contender: str) -> pd.DataFrame:
    rows=[]
    for budget, g in df.groupby('budget'):
        for metric in METRICS:
            a = g[g.method==contender][['seed', metric]].rename(columns={metric:'a'})
            b = g[g.method==baseline][['seed', metric]].rename(columns={metric:'b'})
            m = a.merge(b, on='seed')
            if len(m)==0:
                continue
            if metric in {'aggregate_distortion','false_positive_rate','runtime_sec'}:
                diffs = m['b'].to_numpy() - m['a'].to_numpy()  # positive means contender is better
            else:
                diffs = m['a'].to_numpy() - m['b'].to_numpy()
            rng=np.random.default_rng(777+int(round(float(budget)*1000)))
            lo,hi=bootstrap_ci(diffs,rng)
            rows.append({
                'budget': float(budget), 'baseline': baseline, 'contender': contender,
                'metric': metric, 'paired_mean_improvement': float(np.mean(diffs)),
                'ci95_low': lo, 'ci95_high': hi, 'sign_test_p_greater': sign_test_greater(diffs),
                'n_pairs': int(len(m))
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', default='data/results')
    args=ap.parse_args()
    root=Path(args.results)
    df=pd.read_csv(root/'benchmark_results.csv')
    tests=[]
    for baseline in ['dependency_truth','source_dependence','iterative_truth','provenance_unaware','majority','no_repair']:
        tests.append(paired_tests(df, baseline, 'parc'))
    stat=pd.concat(tests,ignore_index=True)
    stat.to_csv(root/'paired_statistical_tests.csv',index=False)
    # paper table at the budget most often discussed in the text
    for beta in [0.20,0.30,0.35]:
        g=df[np.isclose(df.budget,beta)].groupby('method').agg({
            'integrated_quality':['mean','std'],
            'cell_accuracy':['mean','std'],
            'aggregate_distortion':['mean','std'],
            'false_positive_rate':['mean','std'],
            'runtime_sec':['mean','std'],
            'entity_f1':['mean','std'],
        })
        g.columns=['_'.join(c) for c in g.columns]
        g=g.reset_index().sort_values('integrated_quality_mean',ascending=False)
        g.to_csv(root/f'paper_table_beta{int(beta*100):02d}.csv',index=False)
    # clean headline with CI for PARC and strongest baseline at beta=.35
    beta=0.35
    g=df[np.isclose(df.budget,beta)]
    rows=[]
    rng=np.random.default_rng(42)
    for method in sorted(g.method.unique()):
        gm=g[g.method==method]
        row={'method':method,'budget':beta,'n':len(gm)}
        for metric in ['integrated_quality','aggregate_distortion','false_positive_rate','runtime_sec']:
            vals=gm[metric].to_numpy()
            lo,hi=bootstrap_ci(vals,rng)
            row[f'{metric}_mean']=float(np.mean(vals))
            row[f'{metric}_ci95_low']=lo
            row[f'{metric}_ci95_high']=hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'headline_ci_beta35.csv',index=False)
    print('wrote statistical tables in', root)

if __name__ == '__main__':
    main()
