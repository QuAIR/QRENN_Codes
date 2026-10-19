"""Frozen checkpoint SVC fitting with portable, array-based kernel inference.

The formal API accepts only arm/resample/init. It never fits quantum parameters,
refits on validation, calibrates thresholds, or enables held-out loading.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

# Also constrain numerical libraries when this is the first import in a worker.
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from dataset_screen.data import digest, load_split, write_json
from dataset_screen.evaluate import runtime, save_npz
from eigenbasis_study import evidence as old_evidence
from eigenbasis_study.train import make_tasks as old_tasks
from . import evidence, model

PROTOCOL_PATH=Path(__file__).with_name('protocol.json')
APPROVED_PROTOCOL_SHA256='d138efd3122d01034201830eeb2e8a11215b00120776b12e85773f0ca648ca8a'
PREPROCESSING_ATOL=2*np.finfo(np.float64).eps
METHODS=('fidelity','projected')


def protocol():
    if digest(PROTOCOL_PATH)!=APPROVED_PROTOCOL_SHA256:raise ValueError('Unapproved matched kernel protocol')
    return json.loads(PROTOCOL_PATH.read_text())


def make_tasks():
    p=protocol()
    return [dict(arm=arm,resample=r,init=i) for r in range(p['resamples']) for i in range(p['initializations']) for arm in p['arms']]


def candidate_grid():
    p=protocol()['svm']
    return dict(fidelity=[dict(C=c) for c in p['C']],projected=[dict(C=c,multiplier=g) for c in p['C'] for g in p['multipliers']])


def selection_key(candidate):return (-candidate['metrics']['roc_auc'],candidate['index'])


def _read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError) as exc:raise ValueError('Missing or invalid JSON: '+str(path)) from exc


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _exact(value,expected,label):
    if evidence.array_identity(value)!=expected:raise ValueError('Input/parameter identity mismatch: '+label)


def validate_preprocessing(split,checkpoint,inputs):
    """Only sealed versus raw-derived atan angles may differ by two eps."""
    actual=evidence.input_identity(split)
    for key in ('manifest_sha256','data_sha256','split_sha256'):
        if actual[key]!=inputs[key]:raise ValueError('Data manifest/split identity mismatch: '+key)
    pre=_arrays(Path(checkpoint)/'preprocessing.npz');saved=_arrays(Path(checkpoint)/'preprocessed.npz')
    for key in ('mean','scale'):
        _exact(split[key],inputs[key],key);_exact(pre[key],inputs[key],'saved '+key)
    maximum=0.
    for part in ('fit','validation'):
        section=split[part]
        for key in ('x','standardized','ids','groups','y'):_exact(section[key],inputs[part][key],part+' '+key)
        for key in ('ids','groups','y'):_exact(pre[part+'_'+key],inputs[part][key],'saved '+part+' '+key)
        z=(section['x']-pre['mean'])/pre['scale']
        _exact(z,inputs[part]['standardized'],part+' recomputed standardized')
        angles=.5+np.arctan(z)/np.pi
        if not np.array_equal(angles,section['angles']):raise ValueError('Local raw-derived angle identity mismatch')
        retained=saved[part+'_angles'];_exact(retained,inputs[part]['angles'],'saved '+part+' angles')
        if retained.dtype!=np.dtype('float64') or retained.shape!=angles.shape or not np.isfinite(retained).all():raise ValueError('Invalid sealed angles')
        if not np.allclose(angles,retained,atol=PREPROCESSING_ATOL,rtol=0):raise ValueError('Preprocessing angle drift exceeds tolerance')
        maximum=max(maximum,float(np.max(abs(angles-retained))))
    return maximum


def _checkpoint_path(root,r,i):return Path(root)/'results'/'fixed'/f'r{r}'/f'i{i}'


def _old_config(r,i,pin_hash):
    base=next(c for c in old_tasks() if (c['arm'],c['resample'],c['init'])==('fixed',r,i))
    return dict(base,pin_sha256=pin_hash)


def _original_pin(data_dir,qrenn_root):
    path=Path(qrenn_root)/'pin.json';pin=_read(path)
    if pin.get('schema')!='qrenn-a07-eigenbasis-pin-v1':raise ValueError('Original checkpoint source pin schema mismatch')
    if pin['source_hashes']!=old_evidence.source_hashes():raise ValueError('Original checkpoint source pin mismatch')
    if pin['manifest_sha256']!=digest(Path(data_dir)/'manifest.json'):raise ValueError('Original data pin mismatch')
    if pin['protocol_sha256']!=digest(Path(__file__).parents[1]/'eigenbasis_study/protocol.json'):raise ValueError('Original checkpoint protocol pin mismatch')
    return dict(path=str(path.resolve()),sha256=digest(path))


def _checkpoint_record(data_dir,root,r,i,original_pin,split=None):
    path=_checkpoint_path(root,r,i)
    completion=old_evidence.authenticate(path,expected_config=_old_config(r,i,original_pin['sha256']))
    identity=_read(path/'identity.json')
    from eigenbasis_study.train import protocol as old_protocol
    if (identity['protocol']!=old_protocol() or completion.get('config')!=identity['config']
            or completion.get('actual_epochs')!=1000 or completion.get('early_stopping') is not False):
        raise ValueError('Original fixed-arm completion/protocol inconsistency')
    split=load_split(data_dir,'phishing',r) if split is None else split
    _validate_split(split)
    if split['fit']['x'].shape[1]!=30:raise ValueError('Formal study requires all 30 original features')
    error=validate_preprocessing(split,path,identity['inputs'])
    stages={}
    for stage in ('initial','best'):
        params=_arrays(path/(stage+'.npz'));_validate_parameters(params['theta'],params['probe'],5,formal=True)
        stages[stage]=dict(sha256=digest(path/(stage+'.npz')),arrays={k:evidence.array_identity(params[k]) for k in ('theta','probe')})
    with np.load(path/'validation.npz',allow_pickle=False) as z:
        for key in ('ids','groups','y'):_exact(z[key],identity['inputs']['validation'][key],'original prediction '+key)
        if z['scores'].shape!=split['validation']['y'].shape or not np.isfinite(z['scores']).all():raise ValueError('Invalid original validation predictions')
    return dict(path=str(path.resolve()),completion_sha256=digest(path/'completion.json'),
        identity_sha256=digest(path/'identity.json'),stages=stages,inputs=identity['inputs'],preprocessing_max_error=error)


def create_pin(data_dir,qrenn_root,pin_path):
    """Authenticate all fifteen fixed-arm jobs and bind exact stages before fitting."""
    pin_path=Path(pin_path)
    if pin_path.exists():raise ValueError('Existing source pin preserved')
    p=protocol();source=evidence.source_hashes();original=_original_pin(data_dir,qrenn_root)
    checkpoints={}
    for r in range(5):
        split=load_split(data_dir,'phishing',r)
        for i in range(3):checkpoints[f'r{r}/i{i}']=_checkpoint_record(data_dir,qrenn_root,r,i,original,split)
    if source!=evidence.source_hashes():raise ValueError('Scientific source changed during pinning')
    if digest(original['path'])!=original['sha256']:raise ValueError('Original pin changed during pinning')
    result=dict(schema='qrenn-a07-matched-kernel-pin-v1',data_dir=str(Path(data_dir).resolve()),qrenn_root=str(Path(qrenn_root).resolve()),
        source_hashes=source,protocol_sha256=digest(PROTOCOL_PATH),manifest_sha256=digest(Path(data_dir)/'manifest.json'),
        original_pin=original,checkpoints=checkpoints,protocol=p)
    write_json(pin_path,result)
    return result


def _verify_pin(data_dir,qrenn_root,pin_path):
    pin=_read(pin_path)
    if pin.get('schema')!='qrenn-a07-matched-kernel-pin-v1':raise ValueError('Invalid matched kernel pin schema')
    if pin['source_hashes']!=evidence.source_hashes():raise ValueError('Pinned scientific source mismatch')
    if pin['protocol_sha256']!=digest(PROTOCOL_PATH) or pin['protocol']!=protocol():raise ValueError('Pinned protocol mismatch')
    if (pin['data_dir'],pin['qrenn_root'])!=(str(Path(data_dir).resolve()),str(Path(qrenn_root).resolve())):raise ValueError('Pinned directory identity mismatch')
    if pin['manifest_sha256']!=digest(Path(data_dir)/'manifest.json'):raise ValueError('Pinned manifest mismatch')
    if pin['original_pin']!=_original_pin(data_dir,qrenn_root):raise ValueError('Original source pin changed')
    wanted={f'r{r}/i{i}' for r in range(5) for i in range(3)}
    if set(pin['checkpoints'])!=wanted:raise ValueError('Incomplete declared checkpoint grid')
    for r in range(5):
        for i in range(3):
            record=pin['checkpoints'][f'r{r}/i{i}'];path=_checkpoint_path(qrenn_root,r,i)
            if record['path']!=str(path.resolve()) or digest(path/'completion.json')!=record['completion_sha256']:raise ValueError('Pinned checkpoint grid/path/completion mismatch')
    return pin


def _formal_context(data_dir,qrenn_root,pin_path,config):
    pin=_verify_pin(data_dir,qrenn_root,pin_path)
    split=load_split(data_dir,'phishing',config['resample'])
    record=_checkpoint_record(data_dir,qrenn_root,config['resample'],config['init'],pin['original_pin'],split)
    if record!=pin['checkpoints'][f"r{config['resample']}/i{config['init']}"]:raise ValueError('Pinned checkpoint/input identity changed')
    stage=protocol()['checkpoint_stages'][config['arm']];params=_arrays(Path(record['path'])/(stage+'.npz'))
    refs=dict(pin_path=str(Path(pin_path).resolve()),pin_sha256=digest(pin_path),data_dir=str(Path(data_dir).resolve()),
        qrenn_root=str(Path(qrenn_root).resolve()),checkpoint=record,stage=stage)
    return split,params,refs


def verify_formal_references(identity):
    refs=identity['references']
    if digest(refs['pin_path'])!=refs['pin_sha256']:raise ValueError('Pinned execution identity changed')
    split,params,current=_formal_context(refs['data_dir'],refs['qrenn_root'],refs['pin_path'],identity['config'])
    if current!=refs or evidence.input_identity(split)!=identity['inputs']:raise ValueError('Formal pinned inputs/checkpoint changed')
    for key in ('theta','probe'):_exact(params[key],identity['parameters'][key],key)


def fit(data_dir,qrenn_root,pin_path,out_dir,config):
    """Formal callable: only literal task coordinates, with no scientific overrides."""
    if (not isinstance(config,dict) or set(config)!={'arm','resample','init'} or
            any(type(config.get(k)) is not int for k in ('resample','init')) or config not in make_tasks()):
        raise ValueError('Formal configuration must equal literal arm/resample/init task coordinates')
    split,params,refs=_formal_context(data_dir,qrenn_root,pin_path,config)
    return _run(split,out_dir,params['theta'],params['probe'],config,n=5,candidates=candidate_grid(),mode='formal',references=refs)


def _validate_parameters(theta,probe,n,*,formal=False):
    if type(n) is not int or n<1:raise ValueError('Positive integer data register required')
    for value in (theta,probe):
        if not isinstance(value,np.ndarray) or value.dtype!=np.dtype('float64') or not np.isfinite(value).all():raise ValueError('Exact finite float64 parameters required')
    if theta.ndim!=2 or theta.shape[1]!=15 or probe.shape!=(2**n-1,):raise ValueError('Invalid checkpoint parameter shape')
    T=theta.shape[0]-1
    if formal and (n!=5 or T!=8):raise ValueError('Formal checkpoint requires n5,m2,T8')
    if not formal and (type(n) is not int or not 1<=n<=2 or not 0<=T<=3):raise ValueError('Synthetic bound is total qubits<=4 and T<=3')
    return T


def _validate_split(split):
    evidence.input_identity(split)
    mean=np.asarray(split['mean']);scale=np.asarray(split['scale'])
    if mean.dtype!=np.dtype('float64') or scale.dtype!=mean.dtype or mean.ndim!=1 or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid fit-only scaler')
    for part in ('fit','validation'):
        section=split[part];x=np.asarray(section['x']);y=np.asarray(section['y'])
        if x.dtype!=np.dtype('float64') or x.ndim!=2 or x.shape[1]!=len(mean) or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid original feature matrix')
        z=(x-mean)/scale;angles=.5+np.arctan(z)/np.pi
        if not np.isfinite(z).all():raise ValueError('Nonfinite standardized features')
        for key,value in (('standardized',z),('angles',angles)):
            if evidence.array_identity(section[key])!=evidence.array_identity(value):raise ValueError('Raw-derived preprocessing identity mismatch: '+key)
        if y.dtype.kind not in 'iu' or y.shape!=(len(x),) or set(np.unique(y))!={0,1}:raise ValueError('Both binary classes required')
        for key in ('ids','groups'):
            value=np.asarray(section[key])
            if value.dtype.kind not in 'iu' or value.shape!=y.shape:raise ValueError('Invalid row/group IDs')
        if len(np.unique(section['ids']))!=len(y):raise ValueError('Duplicate row IDs')
    for key in ('ids','groups'):
        if np.intersect1d(split['fit'][key],split['validation'][key]).size:raise ValueError('Fit/validation row/group overlap')


def _validate_candidates(candidates):
    if not isinstance(candidates,dict) or set(candidates)!=set(METHODS):raise ValueError('Both candidate methods required')
    for method,grid in candidates.items():
        if not isinstance(grid,list) or not 1<=len(grid)<=2:raise ValueError('Synthetic bound is one or two candidates per method')
        for config in grid:
            if set(config)!=({'C'} if method=='fidelity' else {'C','multiplier'}):raise ValueError('Unexpected candidate scientific override')
            for value in config.values():
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:raise ValueError('Positive finite candidate values required')


def fit_synthetic(split,out_dir,theta,probe,*,arm,n,candidates):
    """Artificial-only bounded API: <=24 fit, <=16 validation, <=4 total qubits."""
    _validate_split(split);_validate_parameters(theta,probe,n);_validate_candidates(candidates)
    if arm not in ('initial','trained'):raise ValueError('Unknown checkpoint arm')
    if len(split['fit']['y'])>24 or len(split['validation']['y'])>16:raise ValueError('Synthetic row budget exceeded')
    return _run(split,out_dir,theta,probe,dict(arm=arm),n=n,candidates=candidates,mode='synthetic',references=None)


def _metrics(y,scores):
    return dict(roc_auc=float(roc_auc_score(y,scores)),average_precision=float(average_precision_score(y,scores)),
        accuracy=float(accuracy_score(y,scores>=0)),balanced_accuracy=float(balanced_accuracy_score(y,scores>=0)))


def _rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system()=='Darwin' else value*1024)


def _design(method,left,right,gamma):
    return model.fidelity_kernel(left,right) if method=='fidelity' else model.gaussian_kernel(left,right,gamma)


def _run(split,out_dir,theta,probe,config,*,n,candidates,mode,references):
    _validate_split(split);T=_validate_parameters(theta,probe,n,formal=mode=='formal')
    if split['fit']['x'].shape[1]>2**n:raise ValueError('Feature count exceeds data register')
    p=protocol();out=Path(out_dir);started=time.monotonic();cpu=time.process_time();torch.set_num_threads(1)
    parameters={k:evidence.array_identity(v) for k,v in (('theta',theta),('probe',probe))}
    identity=dict(schema='qrenn-a07-matched-kernel-identity-v1',evidence_mode=mode,config=config,protocol=p,
        protocol_sha256=digest(PROTOCOL_PATH),source_hashes=evidence.source_hashes(),runtime=runtime(),
        inputs=evidence.input_identity(split),parameters=parameters,checkpoint=dict(stage=p['checkpoint_stages'][config['arm']]),
        references=references,model=dict(n=n,m=2,T=T),candidates=candidates)
    if mode=='formal':identity['checkpoint'].update(references['checkpoint'])
    def verify():
        if evidence.source_hashes()!=identity['source_hashes']:raise ValueError('Scientific source changed during fitting')
        if evidence.input_identity(split)!=identity['inputs']:raise ValueError('In-memory inputs changed during fitting')
        for k,v in (('theta',theta),('probe',probe)):_exact(v,parameters[k],k)
        if mode=='formal':verify_formal_references(identity)
    verify();existing=evidence.begin(out,identity)
    if existing is not None:return existing
    write_json(out/'protocol.json',p);save_npz(out/'parameters.npz',theta=theta,probe=probe)
    save_npz(out/'preprocessing.npz',mean=split['mean'],scale=split['scale'],**{part+'_'+k:split[part][k] for part in ('fit','validation') for k in ('ids','groups','y')})
    save_npz(out/'fit_inputs.npz',**{k:split['fit'][k] for k in ('x','ids','groups','y')})
    save_npz(out/'preprocessed.npz',**{part+'_angles':split[part]['angles'] for part in ('fit','validation')})
    planned=[dict(method=method,index=i) for method in METHODS for i in range(len(candidates[method]))];finished=[]
    def progress(stage,current=None):
        write_json(out/'progress.json',dict(stage=stage,current=current,planned=planned,finished=finished.copy(),elapsed_seconds=time.monotonic()-started))
    progress('preparing_states')
    with threadpool_limits(limits=1):
        states={part:model.feature_states(split[part]['angles'],theta,probe,n=n) for part in ('fit','validation')}
        features={part:model.projected_features(states[part]) for part in states}
        zz=features['validation'][:,-1].copy();zz_error=None
        if mode=='formal' and config['arm']=='trained':
            reference=_arrays(Path(references['checkpoint']['path'])/'validation.npz')['scores']
            zz_error=float(np.max(abs(zz-reference)))
            if not np.allclose(zz,reference,atol=2e-10,rtol=0):raise ValueError('Reconstructed trained checkpoint ZZ prediction mismatch')
        save_npz(out/'qrenn_zz_validation.npz',scores=zz,**{k:split['validation'][k] for k in ('ids','groups','y')})
        cache_bytes=sum(a.nbytes for a in states.values())+sum(a.nbytes for a in features.values());peak_design=0
        selected={};outcomes=[]
        for method in METHODS:
            representation=states if method=='fidelity' else features
            fit_design=val_design=None;cached_gamma=object();successful=[];models={}
            for index,candidate in enumerate(candidates[method]):
                current=dict(method=method,index=index);progress('before_candidate',current);verify()
                cstarted=time.monotonic();ccpu=time.process_time();gamma=None;resolved=None
                try:
                    gamma=None if method=='fidelity' else model.projected_gamma(features['fit'],candidate['multiplier'])
                    if fit_design is None or gamma!=cached_gamma:
                        # Drop the prior pair before construction; never keep all candidate matrices.
                        fit_design=val_design=None
                        fit_design=_design(method,representation['fit'],representation['fit'],gamma)
                        val_design=_design(method,representation['validation'],representation['fit'],gamma)
                        fit_design.setflags(write=False);val_design.setflags(write=False);cached_gamma=gamma
                    design_bytes=fit_design.nbytes+val_design.nbytes;peak_design=max(peak_design,design_bytes)
                    resolved=dict(C=float(candidate['C']),kernel='precomputed',class_weight='balanced',tol=.001,cache_size=512,max_iter=-1)
                    estimator=SVC(**resolved);estimator.fit(fit_design,split['fit']['y'])
                    values=np.asarray(estimator.decision_function(val_design),dtype=np.float64)
                    if values.shape!=split['validation']['y'].shape or not np.isfinite(values).all():raise ValueError('Invalid SVC validation scores')
                    item=dict(method=method,index=index,candidate=candidate,svc=resolved,gamma=gamma,status='complete',metrics=_metrics(split['validation']['y'],values),support_count=int(len(estimator.support_)))
                    # Retain only compact support representations, never fitted Gram matrices.
                    models[index]=dict(support_indices=estimator.support_.astype(np.int64),dual_coef=estimator.dual_coef_[0].copy(),intercept=estimator.intercept_.copy(),classes=estimator.classes_.copy())
                    successful.append(item)
                    save_npz(out/'candidates'/f'{method}_{index:03d}.npz',scores=values,**{k:split['validation'][k] for k in ('ids','groups','y')})
                    del estimator
                except Exception as exc:
                    item=dict(method=method,index=index,candidate=candidate,svc=resolved,gamma=gamma,status='failed',error_type=type(exc).__name__,error=str(exc))
                    save_npz(out/'candidates'/f'{method}_{index:03d}.npz',scores=np.empty(0,dtype=np.float64))
                item['costs']=dict(elapsed_seconds=time.monotonic()-cstarted,cpu_seconds=time.process_time()-ccpu,peak_process_rss_bytes=_rss())
                write_json(out/'candidates'/f'{method}_{index:03d}.json',item);outcomes.append(item);finished.append(current);progress('after_candidate',current);verify()
            if successful:
                winner=min(successful,key=selection_key);index=winner['index'];compact=models[index]
                save_npz(out/'selected'/f'{method}.npz',**compact,config_json=np.array(json.dumps(winner,sort_keys=True)))
                values=_arrays(out/'candidates'/f'{method}_{index:03d}.npz')
                save_npz(out/'validation'/f'{method}.npz',**values);selected[method]=winner
            fit_design=val_design=None
        failed=sum(item['status']=='failed' for item in outcomes)
        if failed:
            progress('failed');raise ValueError(f'{failed} candidate failures retained; no successful completion sealed')
    recipe=dict(schema='qrenn-a07-matched-kernel-recipe-v1',source_hashes=identity['source_hashes'],parameters='parameters.npz',
        preprocessing='preprocessing.npz',fit_inputs='fit_inputs.npz',selected=selected,model=identity['model'],
        threshold=0.,refit=False,native_pickle_required=False,original_feature_inference=True)
    write_json(out/'recipe.json',recipe);verify();progress('complete')
    costs=dict(elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,peak_process_rss_bytes=_rss(),
        prepared_cache_bytes=cache_bytes,peak_design_array_bytes=peak_design,peak_accounted_array_bytes=cache_bytes+peak_design,
        torch_threads=1,blas_threads=1,device='cpu')
    write_json(out/'costs.json',costs)
    resources=dict(num_data_qubits=n,num_processing_qubits=2,total_qubits=n+2,state_preparation_queries=T,overlap_queries=2*T,
        projected_feature_count=features['fit'].shape[1],measurement_bases_up_to=3,no_equal_total_hardware_cost_claim=True)
    verify()
    return evidence.seal(out,dict(config=config,selected=selected,candidate_accounting=dict(planned=len(planned),finished=len(finished),successful=len(outcomes)-failed,failed=failed),
        resource=resources,costs=costs,trained_zz_max_error=zz_error,initial_zz_scope='raw diagnostic only; not trained accuracy' if config['arm']=='initial' else None,
        preprocessing_max_error=None if references is None else references['checkpoint']['preprocessing_max_error']))


def predict_original_features(out_dir,x,method,*,evidence_mode='formal'):
    """Portable SVC score = support kernel @ signed dual coefficients + intercept."""
    out=Path(out_dir);evidence.authenticate(out,evidence_mode=evidence_mode)
    if method not in METHODS:raise ValueError('Unknown kernel method')
    recipe=_read(out/'recipe.json');pre=_arrays(out/'preprocessing.npz');params=_arrays(out/'parameters.npz');fitting=_arrays(out/'fit_inputs.npz');selected=_arrays(out/'selected'/f'{method}.npz')
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(pre['mean']) or not np.isfinite(x).all():raise ValueError('Invalid original features')
    if not len(x):return np.empty(0,dtype=np.float64)
    support=fitting['x'][selected['support_indices']]
    def states(raw):return model.feature_states(.5+np.arctan((raw-pre['mean'])/pre['scale'])/np.pi,params['theta'],params['probe'],n=recipe['model']['n'])
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        left,right=states(x),states(support)
        if method=='projected':left,right=model.projected_features(left),model.projected_features(right)
        design=_design(method,left,right,recipe['selected'][method]['gamma'])
        return design@selected['dual_coef']+float(selected['intercept'][0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--qrenn',type=Path,required=True)
    parser.add_argument('--create-pin',type=Path);parser.add_argument('--pin',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--arm',choices=('initial','trained'));parser.add_argument('--r',type=int);parser.add_argument('--i',type=int)
    args=parser.parse_args()
    if args.create_pin is not None:
        if any(v is not None for v in (args.pin,args.out,args.arm,args.r,args.i)):parser.error('create-pin cannot be combined with fitting coordinates')
        result=create_pin(args.data,args.qrenn,args.create_pin)
        print(json.dumps(dict(status='pinned',checkpoints=len(result['checkpoints']))))
    else:
        if any(v is None for v in (args.pin,args.out,args.arm,args.r,args.i)):parser.error('fitting requires pin, out, arm, r and i')
        result=fit(args.data,args.qrenn,args.pin,args.out,dict(arm=args.arm,resample=args.r,init=args.i))
        print(json.dumps(dict(status='complete',candidate_accounting=result['candidate_accounting'])))


if __name__=='__main__':main()
