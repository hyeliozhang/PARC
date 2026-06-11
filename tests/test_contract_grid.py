import pandas as pd


def test_contract_grid_contains_six_datasets_and_two_budgets():
    df = pd.read_csv('data/results/contract_grid_results.csv')
    assert set(df['dataset']) >= {'synthetic','iris','wine','breast_cancer','diabetes','digits'}
    assert set(df['budget'].round(2)) >= {0.20}
    assert {'no_repair','dependency_truth','parc'}.issubset(set(df['method']))


def test_parc_contract_grid_is_fair_against_baseline_certificate_at_public_budget20():
    df = pd.read_csv('data/results/contract_grid_paper.csv')
    pub = df[(df['suite']=='public-tabular') & (df['budget'].round(2)==0.20)]
    parc = pub[pub['method']=='parc'].iloc[0]
    dep = pub[pub['method']=='dependency_truth'].iloc[0]
    assert parc['coverage_mean'] >= 0.90
    assert parc['miss_distance_mean'] <= dep['miss_distance_mean'] + 1e-9
    assert parc['point_distortion_mean'] <= 0.01
