"""Bounded end-to-end development/freeze/test/replay fixtures; no formal fits."""
import json
from pathlib import Path
import numpy as np
import pytest
from . import kernels,baselines,evaluate,campaign,evidence,train,data

@pytest.fixture
def split():
    rng=np.random.default_rng(173);x=rng.normal(size=(28,4));y=(x[:,0]+.4*x[:,1]>0).astype(np.int64)
    # Balanced labels avoid degenerate tiny fixture metrics.
    y[np.arange(0,28,2)]=0;y[np.arange(1,28,2)]=1
    mean=x[:16].mean(0);scale=x[:16].std(0)
    result=dict(mean=mean,scale=scale,manifest_sha256='fixture',data_sha256='fixture',split_sha256='fixture')
    for name,ids in [('fit',np.arange(16)),('validation',np.arange(16,22)),('test',np.arange(22,28))]:
        z=(x[ids]-mean)/scale
        result[name]=dict(x=x[ids],standardized=z,angles=.5+np.arctan(z)/np.pi,y=y[ids],ids=ids,groups=ids)
    return result

def development(split):return {k:v for k,v in split.items() if k!='test'}

def test_frozen_grids_and_campaign_guard(tmp_path):
    assert len(campaign.tasks())==100
    assert len(evaluate.expected_coordinates())==100
    assert {m:len(v) for m,v in kernels.grid().items()}==dict(fidelity=4,projected=12)
    assert {m:len(v) for m,v in baselines.grid(8).items()}==dict(lr_standardized=5,lr_angles=5,rbf_svm=12,gradient_boosting=12)
    with pytest.raises(ValueError):evaluate.evaluate(tmp_path,tmp_path/'absent',tmp_path/'evaluation')


def test_all_models_freeze_prediction_replay_and_no_refit(split,tmp_path,monkeypatch):
    dev=development(split);cfg=dict(dataset='htru2',resample=0,init=0)
    training=dict(arm='fixed',T=2,n=2,scale=1.,seed=82000,shuffle_seed=1082000,device='cpu',**cfg)
    checkpoint=tmp_path/'training';train.fit_synthetic(dev,checkpoint,training,epochs=1,batch_size=8)
    q=tmp_path/'qrenn';kernels.import_qrenn_synthetic(dev,q,checkpoint,cfg)
    params=kernels.arrays(checkpoint/'initial.npz');bundles=[q]
    for arm in ('initial','trained'):
        params=kernels.arrays(checkpoint/('initial.npz' if arm=='initial' else 'best.npz'));out=tmp_path/arm
        kernels.fit_synthetic(dev,out,params['theta'],params['probe'],dict(cfg,arm=arm));bundles.append(out)
    classical=tmp_path/'classical';baselines.fit_synthetic(dev,classical,dict(dataset='htru2',resample=0));bundles.append(classical)
    assert all(evaluate.check_candidates(p,mode='synthetic') for p in bundles)
    with pytest.raises(ValueError,match='All 60|source|Pinned|Invalid|Missing|No such|schema'):
        evaluate.freeze(tmp_path,bundles,tmp_path/'formal.json',pin_path=tmp_path/'missing')
    snapshot=tmp_path/'freeze.json';frozen=evaluate.freeze_synthetic(bundles,snapshot);assert frozen['selected_count']==9
    before={str(p):evidence.inventory(p) for p in bundles}
    # Evaluation must remain possible with every estimator fit method disabled.
    def forbidden(*args,**kwargs):raise AssertionError('Evaluation refitted a model')
    monkeypatch.setattr(baselines.LogisticRegression,'fit',forbidden);monkeypatch.setattr(baselines.SVC,'fit',forbidden);monkeypatch.setattr(baselines.HistGradientBoostingClassifier,'fit',forbidden)
    result=evaluate.evaluate_synthetic(snapshot,tmp_path/'evaluation',{('htru2',0):split})
    assert result['selected_model_count']==9 and len(result['summary'])==18
    assert {row['method'] for row in result['summary']}=={'qrenn_zz','initial_fidelity_svm','initial_projected_svm','trained_fidelity_svm','trained_projected_svm',*baselines.METHODS}
    assert all(row['average_precision_sample_sd'] is None for row in result['summary'])
    audits=kernels.read(tmp_path/'evaluation'/'replay_audit.json');assert len(audits)==18
    assert max(a['max_abs_score_error'] for a in audits)<2e-10
    assert evaluate.audit_evaluation(tmp_path/'evaluation')['predictions']==18
    assert before=={str(p):evidence.inventory(p) for p in bundles}
    (classical/'selected'/'lr_angles.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='mutated'):evaluate.authenticate_snapshot(snapshot,mode='synthetic')


def test_test_exclusion_and_formal_freeze_incomplete(split,tmp_path):
    theta=np.zeros((2,15));probe=np.zeros(3);cfg=dict(dataset='magic',resample=0,init=0,arm='initial')
    with pytest.raises(ValueError,match='exclude test'):kernels.fit_synthetic(split,tmp_path/'bad',theta,probe,cfg)
    with pytest.raises(ValueError,match='exclude test'):baselines.fit_synthetic(split,tmp_path/'bad2',dict(dataset='magic',resample=0))
    assert not (tmp_path/'bad').exists() and not (tmp_path/'bad2').exists()


def test_independent_generators_and_rank_reporting():
    from matched_kernel_study import model
    rng=np.random.default_rng(9)
    for d in (8,10,57):
        x=rng.normal(size=(3,d));mean=np.zeros(d);scale=np.ones(d);theta=rng.normal(size=(9,15));probe=rng.normal(size=(2**((d-1).bit_length())-1))
        independent=evaluate.numpy_states(x,mean,scale,theta,probe)
        production=model.feature_states(.5+np.arctan(x)/np.pi,theta,probe,n=(d-1).bit_length())
        np.testing.assert_allclose(independent,production,atol=2e-12,rtol=0)
        np.testing.assert_allclose(evaluate.numpy_pauli(independent),model.projected_features(production),atol=2e-12,rtol=0)
    discrepancy=evaluate.discrepancy(np.array([0.,1e-15,1.]),np.array([1e-15,0.,1.]))
    assert discrepancy['ranking_disagreements']==2 and discrepancy['predictions_rounded'] is False


def test_hierarchical_aggregation_and_ties():
    rows=[]
    for r in range(5):
        for i in range(3):rows.append(dict(partition='test',dataset='magic',resample=r,init=i,method='qrenn_zz',roc_auc=r+i,average_precision=r+i,accuracy=r+i,balanced_accuracy=r+i))
        rows.append(dict(partition='test',dataset='magic',resample=r,init=None,method='gradient_boosting',roc_auc=r,average_precision=r,accuracy=r,balanced_accuracy=r))
    result=evaluate.aggregate(rows);q=next(r for r in result['summary'] if r['method']=='qrenn_zz')
    assert q['roc_auc_mean']==3 and q['roc_auc_sample_sd']==pytest.approx(np.std(np.arange(5),ddof=1))
    assert all(abs(v['roc_auc'])==1 for v in result['paired_differences'])
    candidates=[dict(index=1,metrics=dict(roc_auc=.9,average_precision=.8)),dict(index=0,metrics=dict(roc_auc=.8,average_precision=.9))]
    assert min(candidates,key=lambda v:kernels.selection_key(v,'htru2'))['index']==0
    assert min(candidates,key=lambda v:kernels.selection_key(v,'magic'))['index']==1


@pytest.mark.parametrize('arm',['initial','trained'])
def test_official_phishing_kernel_import(tmp_path,arm,monkeypatch):
    literal=data.protocol
    monkeypatch.setattr(data,'protocol',lambda:dict(literal(),datasets=['magic','htru2','phishing','spambase']))
    import os
    root=Path(os.environ.get('QRENN_EVIDENCE_ROOT','/Users/mingrui/Projects/QRENN/output'))
    official=root/'prxq-a07-multidata-matched-2026-10-18';old=root/'prxq-a07-matched-kernel-2026-10-18/evidence'
    prepared=tmp_path/'data';data.prepare(official/'raw',prepared)
    cfg=dict(dataset='phishing',resample=0,init=0,arm=arm);bundle=tmp_path/'kernel'
    completion=kernels.import_phishing(prepared,bundle,old/('campaign/results/'+arm+'/r0/i0'),cfg,legacy_data=root/'prxq-a07-dataset-screen-2026-10-17/data',legacy_qrenn=old/'dependencies/qrenn')
    if arm=='trained':
        qbundle=tmp_path/'qrenn';qconfig={key:cfg[key] for key in ('dataset','resample','init')}
        kernels.import_qrenn(prepared,qbundle,None,qconfig,legacy_data=root/'prxq-a07-dataset-screen-2026-10-17/data',legacy_qrenn=old/'dependencies/qrenn')
        assert evaluate.check_candidates(qbundle)['history_count']==201
    assert completion['reused_source'] and completion['candidate_count']==16
    audit=evaluate.check_candidates(bundle);assert len(audit['checks'])==2
    split=data.load_split(prepared,'phishing',0)
    for method in kernels.METHODS:
        # Only bounded inference rows: no old/new model fitting occurs.
        raw=split['validation']['x'][:5];saved=kernels.arrays(bundle/('validation_'+method+'.npz'))['scores'][:5]
        np.testing.assert_allclose(kernels.predict_original_features(bundle,raw,method),saved,atol=2e-10,rtol=0)
        np.testing.assert_allclose(evaluate.independent_predict(bundle,raw,method),saved,atol=2e-10,rtol=0)


def test_portable_numeric_boosting_nontrivial_trees():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng=np.random.default_rng(779);x=rng.normal(size=(96,4));y=(x[:,0]+x[:,1]>.1).astype(np.int64)
    estimator=HistGradientBoostingClassifier(max_iter=8,learning_rate=.1,max_leaf_nodes=7,min_samples_leaf=3,early_stopping=False,random_state=39011).fit(x,y)
    compact=baselines._portable(estimator,'gradient_boosting',x)
    assert any(np.count_nonzero(nodes['is_leaf']==0)>0 for name,nodes in compact.items() if name.startswith('tree_'))
    replay=np.full(len(x),compact['baseline'][0])
    for name,nodes in compact.items():
        if name.startswith('tree_'):replay+=baselines.tree_scores(nodes,x)
    np.testing.assert_allclose(replay,estimator.decision_function(x),atol=2e-15,rtol=0)
