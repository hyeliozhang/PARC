#!/usr/bin/env python3
"""CPU-light held-out public-tabular transfer evaluation without SQL execution."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_external_and_sql_suite import evaluate_real_dataset

def summarize(df, cols):
    nums=[c for c in df.columns if c not in set(cols) and pd.api.types.is_numeric_dtype(df[c])]
    out=df.groupby(cols,as_index=False).agg({c:['mean','std','min','max'] for c in nums})
    out.columns=['_'.join([p for p in col if p]) for col in out.columns.to_flat_index()]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT/'data'/'results'))
    ap.add_argument('--datasets', nargs='+', default=['iris','wine','breast_cancer','diabetes','digits','mpg','txhousing','midwest'])
    ap.add_argument('--budgets', nargs='+', type=float, default=[0.17,0.27])
    ap.add_argument('--seeds', nargs='+', type=int, default=[41,42])
    ap.add_argument('--methods', nargs='+', default=['no_repair','dependency_truth','parc'])
    args=ap.parse_args()
    rows=[]
    for ds in args.datasets:
        for b in args.budgets:
            for seed in args.seeds:
                rows.extend(evaluate_real_dataset(ds,b,seed,args.methods))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'heldout_public_no_sql_results.csv',index=False)
    summarize(df,['dataset','budget','method']).to_csv(out/'heldout_public_no_sql_summary.csv',index=False)
    summarize(df,['budget','method']).to_csv(out/'heldout_public_no_sql_overall.csv',index=False)
    print(f'wrote {len(df)} rows to {out}')
if __name__=='__main__': main()
