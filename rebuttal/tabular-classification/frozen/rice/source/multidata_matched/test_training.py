"""Bounded synthetic and official preparation tests; no production model fits."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from . import data, evidence, train
from eigenbasis_study import model
from dataset_screen.model import initial_parameters, scores

OUTPUT=Path(os.environ.get('QRENN_EVIDENCE_ROOT','/Users/mingrui/Projects/QRENN/output'))

@pytest.fixture
def split():
    rng=np.random.default_rng(47);x=rng.normal(size=(18,4));mean=x[:12].mean(0);scale=x[:12].std(0)
    result=dict(mean=mean,scale=scale,manifest_sha256='fixture',data_sha256='fixture',split_sha256='fixture')
    for name,start,end in [('fit',0,12),('validation',12,18)]:
        ids=np.arange(start,end);z=(x[ids]-mean)/scale
        result[name]=dict(x=x[ids],standardized=z,angles=.5+np.arctan(z)/np.pi,y=ids%2,ids=ids,groups=ids)
    return result

CFG=dict(arm='fixed',T=2,n=2,scale=1.,seed=82000,shuffle_seed=1082000,device='cpu',dataset='htru2')

def test_grid_selection_and_group_policy():
    assert len(train.make_tasks())==30
    assert {t['dataset'] for t in train.make_tasks()}=={'shoppers','rice'}
    assert train.primary_metric('htru2')=='average_precision'
    assert train.primary_metric('magic')=='roc_auc'
    assert min([(.8,.2,5),(.8,.1,10),(.8,.1,5),(.7,.01,0)],key=lambda x:train.checkpoint_key(*x))==(.8,.1,5)
    groups,counts,pos,majority,conflicts=data._groups(np.array([[1],[1],[2],[2],[2]]),np.array([0,1,0,1,1]))
    assert list(counts)==[2,3] and list(majority)==[0,1] and conflicts.all()

@pytest.mark.parametrize('features',[7,8,10,57,74])
def test_dimensions_selected_scores_gradients(features):
    torch.set_num_threads(1);a=torch.tensor(np.random.default_rng(2).uniform(size=(3,features)),dtype=torch.float64)
    theta,probe=initial_parameters(features,'diagonal_joint',8,82000)
    theta.requires_grad_();probe.requires_grad_()
    prepared=model.prepare_inputs(a,'fixed',n=(features-1).bit_length())
    selected=model.scores_prepared(model.subset_inputs(prepared,[0,2]),theta,probe)
    direct=scores(a[[0,2]],theta,probe,'diagonal_joint',1.)
    torch.testing.assert_close(selected,direct,atol=2e-12,rtol=0)
    from matched_kernel_study.model import feature_states
    states=feature_states(a[[0,2]].numpy(),theta.detach().numpy(),probe.detach().numpy(),n=(features-1).bit_length())
    zz=(abs(states.reshape(2,4,-1))**2).sum(-1)@np.array([1.,-1.,-1.,1.])
    np.testing.assert_allclose(selected.detach().numpy(),zz,atol=2e-12,rtol=0)
    # Independent full-state central difference covers a processing and a probe coordinate.
    for parameter,coordinate,gradient in [(theta,(0,0),0),(probe,(0,),1)]:
        eps=1e-5;original=float(parameter[coordinate].detach());answers=[]
        for sign in [-1,1]:
            t=theta.detach().numpy().copy();p=probe.detach().numpy().copy()
            (t if gradient==0 else p)[coordinate]=original+sign*eps
            state=feature_states(a[[0,2]].numpy(),t,p,n=(features-1).bit_length())
            answers.append(((abs(state.reshape(2,4,-1))**2).sum(-1)@np.array([1.,-1.,-1.,1.])).sum())
        derivative=(answers[1]-answers[0])/(2*eps)
        analytic=torch.autograd.grad(selected.sum(),(theta,probe),retain_graph=True)[gradient][coordinate]
        assert derivative==pytest.approx(float(analytic),abs=2e-9)

    g1=torch.autograd.grad(selected.sum(),(theta,probe));g2=torch.autograd.grad(direct.sum(),(theta,probe))
    for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=2e-11,rtol=0)


def test_resume_replay_and_mutation(split,tmp_path):
    full=train.fit_synthetic(split,tmp_path/'full',CFG,epochs=6,batch_size=5)
    train.fit_synthetic(split,tmp_path/'resume',CFG,epochs=6,batch_size=5,stop_after_epoch=3)
    resumed=train.fit_synthetic(split,tmp_path/'resume',CFG,epochs=6,batch_size=5,resume=True)
    assert full['best_epoch']==resumed['best_epoch']
    for stage in ('initial','best','terminal'):
        with np.load(tmp_path/'full'/f'{stage}.npz') as a,np.load(tmp_path/'resume'/f'{stage}.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    replay=train.predict_original_features(tmp_path/'full',split['validation']['x'],evidence_mode='synthetic')
    with np.load(tmp_path/'full'/'validation.npz') as v:np.testing.assert_allclose(replay,v['scores'],atol=1e-13)
    history=json.loads((tmp_path/'full'/'history.json').read_text())
    expected=min(history,key=lambda h:train.checkpoint_key(h['validation']['average_precision'],h['validation_loss'],h['epoch']))
    assert full['best_epoch']==expected['epoch']
    (tmp_path/'full'/'history.json').write_text('[]')
    with pytest.raises(ValueError,match='mutated'):evidence.authenticate(tmp_path/'full',evidence_mode='synthetic')


def test_test_refusal(split,tmp_path):
    split['test']=split['validation']
    with pytest.raises(ValueError,match='exclude test'):train.fit_synthetic(split,tmp_path/'bad',CFG,epochs=0)
    assert not (tmp_path/'bad').exists()


def test_official_prepare_and_authentication(tmp_path,monkeypatch):
    literal=data.protocol
    monkeypatch.setattr(data,'protocol',lambda:dict(literal(),datasets=['magic','htru2','phishing','spambase']))
    raw=OUTPUT/'prxq-a07-multidata-matched-2026-10-18/raw';legacy=OUTPUT/'prxq-a07-dataset-screen-2026-10-17/data'
    manifest=data.prepare(raw,tmp_path)
    assert len(manifest['splits'])==20
    assert data.verify_legacy_splits(tmp_path,legacy)
    for dataset in ('magic','htru2','phishing','spambase'):
        spec=data.DATASETS[dataset]
        s=data.load_split(tmp_path,dataset,0);assert 'test' not in s
        allrows=data.load_split(tmp_path,dataset,0,include_test=True)
        assert sum(len(allrows[k]['y']) for k in ('fit','validation','test'))==spec['rows']
        for left,right in [('fit','validation'),('fit','test'),('validation','test')]:
            assert not np.intersect1d(allrows[left]['groups'],allrows[right]['groups']).size
        np.testing.assert_allclose(s['fit']['standardized'].mean(0),0,atol=2e-12)
    pin=tmp_path/'pin.json';train.create_pin(tmp_path,pin);train._verify_pin(tmp_path,pin)
    with pytest.raises(ValueError):train.create_pin(tmp_path,pin)
    (tmp_path/'magic-split-0.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):data.load_split(tmp_path,'magic',0)


def test_official_phishing_adapter_and_raw_protocol_mutation(tmp_path,monkeypatch):
    literal=data.protocol
    monkeypatch.setattr(data,'protocol',lambda:dict(literal(),datasets=['magic','htru2','phishing','spambase']))
    import shutil
    raw=OUTPUT/'prxq-a07-multidata-matched-2026-10-18/raw'
    stage=tmp_path/'raw';shutil.copytree(raw,stage)
    prepared=tmp_path/'data';data.prepare(stage,prepared)
    legacy=OUTPUT/'prxq-a07-dataset-screen-2026-10-17/data'
    root=OUTPUT/'prxq-a07-matched-kernel-2026-10-18/evidence/dependencies/qrenn'
    source=train.phishing_source(prepared,legacy,root,0,0)
    assert source['new_run'] is False and set(source['parameters'])=={'initial','best','terminal'}
    assert source['source']['completion_sha256'] and source['source']['stages']['best']['sha256']
    manifest=prepared/'manifest.json';value=json.loads(manifest.read_text());value['protocol_sha256']='changed';manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='protocol'):data.load_split(prepared,'phishing',0)
    value['protocol_sha256']=data.digest(Path(data.__file__).with_name('protocol.json'));manifest.write_text(json.dumps(value))
    (stage/'327.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw archive'):data.load_split(prepared,'phishing',0)


@pytest.mark.parametrize('failure_epoch',[0,5,6])
def test_unpublished_generation_resume(split,tmp_path,monkeypatch,failure_epoch):
    """Crash after assessment and snapshot writes, before progress publication."""
    control=tmp_path/'control';attempt=tmp_path/'attempt'
    expected=train.fit_synthetic(split,control,CFG,epochs=6,batch_size=5)
    original=train.write_json;failed=[]
    def interrupt(path,value):
        if Path(path)==attempt/'progress.json' and value['epoch']==failure_epoch and not failed:
            failed.append(True)
            raise RuntimeError('Injected failure before progress publication')
        original(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(train,'write_json',interrupt)
        with pytest.raises(RuntimeError,match='Injected failure'):
            train.fit_synthetic(split,attempt,CFG,epochs=6,batch_size=5)
    assert len(list(attempt.glob('checkpoint-*.pt')))==min(failure_epoch+1,3)
    if failure_epoch==0:
        # No published generation exists: evidence remains, restart must use a new directory.
        with pytest.raises(ValueError,match='No authenticated resumable checkpoint'):
            train.fit_synthetic(split,attempt,CFG,epochs=6,batch_size=5,resume=True)
        return
    published=json.loads((attempt/'progress.json').read_text())
    assert published['epoch']==failure_epoch-1
    state=torch.load(attempt/published['checkpoint_file'],weights_only=False)
    assert state['epoch']==published['epoch'] and state['best_parameters']
    assert max(h['epoch'] for h in state['history'])<=published['epoch']
    resumed=train.fit_synthetic(split,attempt,CFG,epochs=6,batch_size=5,resume=True)
    assert resumed['best_epoch']==expected['best_epoch']
    assert len(list(attempt.glob('checkpoint-*.pt')))==2
    assert len(list((attempt/'interrupted').glob('checkpoint-*.pt')))==1  # orphan retained once
    for stage in ('best','terminal'):
        with np.load(attempt/f'{stage}.npz') as a,np.load(control/f'{stage}.npz') as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert json.loads((attempt/'history.json').read_text())==json.loads((control/'history.json').read_text())


@pytest.mark.parametrize('change',[dict(n=7),dict(T=9)])
def test_synthetic_dimension_caps(split,tmp_path,change):
    with pytest.raises(ValueError,match='register/query bounds'):
        train.fit_synthetic(split,tmp_path/'bad',dict(CFG,**change),epochs=0)
    assert not (tmp_path/'bad').exists()
