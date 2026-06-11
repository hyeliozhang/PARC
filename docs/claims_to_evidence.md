# Claims To Evidence

This note maps the public artifact files to the main empirical claims.

| Claim family | Evidence files | Reproduction entry points |
| --- | --- | --- |
| Main repair accuracy and distortion | `data/results/benchmark_results.csv`, `data/results/benchmark_summary.csv`, `data/results/headline_ci_beta35.csv` | `scripts/run_benchmark.py`, `scripts/statistical_analysis.py` |
| Ablations and stress modes | `data/results/ablation_results.csv`, `data/results/operator_ablation_results.csv`, `data/results/stress_modes_results.csv` | `scripts/run_full_benchmark.py`, `scripts/run_adversarial_search.py` |
| Contract and certificate behavior | `data/results/contract_grid_paper.csv`, `data/results/contract_totalnorm_summary.csv`, `data/results/certified_query_bounds_paper.csv` | `scripts/run_contract_grid.py`, `scripts/run_contract_totalnorm.py`, `scripts/certified_query_bounds.py` |
| SQL and external-tabular robustness | `data/results/external_tabular_results.csv`, `data/results/sql_workflow_results.csv`, `data/results/sql_fragment_results.csv` | `scripts/run_external_and_sql_suite.py`, `scripts/run_sql_fragment_suite.py` |
| Scalability and memory | `data/results/scalability_results.csv`, `data/results/runtime_memory_profile.csv`, `data/results/source_scale_parc_summary.csv` | `scripts/runtime_memory_profile.py`, `scripts/run_full_benchmark.py` |

The public repository intentionally excludes manuscript source and submission PDFs. This evidence map is the repository-level bridge from claims to reproducible files.

