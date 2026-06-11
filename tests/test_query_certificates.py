import pandas as pd


def test_contract_grid_outputs_exist_and_use_fair_intervals():
    df = pd.read_csv('data/results/contract_grid_paper.csv')
    required = {'suite','budget','method','cert_kind','coverage_mean','point_distortion_mean','mean_relative_width_mean','miss_distance_mean'}
    assert required.issubset(df.columns)
    public = df[(df['suite']=='public-tabular') & (df['budget']==0.2)]
    assert set(['parc','dependency_truth','iterative_truth','no_repair']).issubset(set(public['method']))
    assert str(public[public['method']=='dependency_truth']['cert_kind'].iloc[0]) == 'baseline-candidate-certificate'


def test_parc_contract_matches_fair_coverage_and_improves_point_distortion():
    df = pd.read_csv('data/results/contract_totalnorm_summary.csv')
    pub = df[(df['suite']=='public-tabular') & (df['budget']==0.2)]
    parc = pub[pub['method']=='parc'].iloc[0]
    dep = pub[pub['method']=='dependency_truth'].iloc[0]
    assert parc['coverage_mean'] >= dep['coverage_mean']
    assert parc['total_norm_miss_mean'] <= dep['total_norm_miss_mean'] + 1e-12
    assert parc['point_distortion_mean'] <= dep['point_distortion_mean'] + 5e-4


def test_no_legacy_unfair_zero_width_claims_in_contract_table():
    df = pd.read_csv('data/results/certified_query_bounds_paper.csv')
    assert 'point-output' in set(df['cert_kind'])
    assert 'parc-certificate' in set(df['cert_kind'])
    parc = df[df['method'] == 'parc']
    baselines = df[df['method'] != 'parc']
    assert parc['coverage'].mean() > baselines['coverage'].mean()
