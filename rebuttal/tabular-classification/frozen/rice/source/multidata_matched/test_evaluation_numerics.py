"""Bounded recovery policy and frozen-evaluator integration tests; synthetic only."""
from pathlib import Path
import importlib
import json
import numpy as np
import pytest
from multidata_matched import evaluate as frozen, evidence, kernels, baselines, train
from multidata_matched.test_comparison import split, development


def helper():
    from . import evaluation_numerics
    return evaluation_numerics


def test_policy_and_cancellation():
    m=helper(); y=np.array([0,1,0,1]); native=np.array([-1.,.5,-.2,.8]); replay=native+6.92e-10
    audit=m.compare_scores(y,native,replay,kind='kernel',method='projected')
    assert audit['original_score_atol']==2e-10 and audit['accepted_score_atol']==2e-8
    assert not audit['original_policy_passed'] and audit['accepted']
    assert audit['classification_disagreements']==audit['ranking_disagreements']==0
    assert set(audit['metric_differences'])==set(train.protocol()['evaluation']['metrics'])
    assert all(v==0 for v in audit['metric_differences'].values())
    assert audit['predictions_rounded'] is False
    assert m.compare_scores(y,native,replay,kind='classical',method='rbf_svm')['accepted']
    for method in ('lr_standardized','lr_angles','gradient_boosting'):
        assert not m.compare_scores(y,native,replay,kind='classical',method=method)['accepted']
    assert not m.compare_scores(y,native,replay,kind='qrenn',method='qrenn_zz')['accepted']


def test_reject_large_classification_and_nonfinite():
    m=helper(); y=np.array([0,1]); reference=np.array([-1e-12,1.])
    for replay in (reference+3e-8, np.array([1e-12,1.])):
        audit=m.compare_scores(y,reference,replay,kind='kernel',method='fidelity')
        assert not audit['accepted']
        with pytest.raises(ValueError):m.require_accepted(audit)
    with pytest.raises(ValueError):m.compare_scores(y,np.array([np.nan,1.]),reference,kind='kernel',method='fidelity')


@pytest.mark.parametrize('kind,method',[('kernel','projected'),('classical','rbf_svm')])
def test_rank_and_four_metric_differences_visible(kind,method):
    m=helper(); y=np.array([0,1,0,1]); reference=np.array([.1,.1+1e-10,-1.,1.]); replay=reference.copy();replay[:2]=replay[:2][::-1]
    audit=m.compare_scores(y,reference,replay,kind=kind,method=method)
    assert audit['accepted'] and audit['ranking_disagreements']==2
    assert [item['row_index'] for item in audit['ranking_differences']]==[0,1]
    assert audit['metric_differences']['roc_auc']!=0 and audit['metric_differences']['average_precision']!=0
    assert audit['metric_differences']['accuracy']==audit['metric_differences']['balanced_accuracy']==0



def test_predeclared_campaign_policy():
    m=helper();p=m.policy()
    assert p['declaration']=='predeclared_before_new_campaign_fitting'
    assert p['relative_tolerance']==0 and p['classification_disagreements_allowed']==0
    assert 'expected_original_snapshot_sha256' not in p and 'frozen_source_commit' not in p
    assert p['previous_campaign_precedent']['source_commit']=='55ac5e0'
    assert p['svm_score_atol']==train.protocol()['evaluation']['svm_score_atol']
