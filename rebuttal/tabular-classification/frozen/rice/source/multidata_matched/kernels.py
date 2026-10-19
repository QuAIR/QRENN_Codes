"""Development-only frozen-circuit kernels and portable source adapters.

Every bundle is self-contained: original fitting rows, scaler, circuit parameters,
all validation candidates and compact selected support coefficients. No pickle or
Gram matrix is needed for prediction. Test arrays are rejected at the API boundary.
"""
from __future__ import annotations
import json
import math
import platform
import resource
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from matched_kernel_study import model
from dataset_screen.evaluate import save_npz
from . import data, evidence, train

METHODS=('fidelity','projected')

def read(path):
    try:return json.loads(Path(path).read_text())
    except (ValueError,OSError) as exc:raise ValueError('Missing or invalid JSON: '+str(path)) from exc
def arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def metrics(y, scores):
    scores=np.asarray(scores)
    if scores.shape!=np.asarray(y).shape or not np.isfinite(scores).all(): raise ValueError('Invalid prediction scores')
    return dict(roc_auc=float(roc_auc_score(y,scores)),average_precision=float(average_precision_score(y,scores)),
                accuracy=float(accuracy_score(y,scores>=0)),balanced_accuracy=float(balanced_accuracy_score(y,scores>=0)))

def selection_key(item,dataset): return (-item['metrics'][train.primary_metric(dataset)],item['index'])

def costs(start,cpu):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                peak_process_rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024),torch_threads=1,blas_threads=1)

def validate(split,mode):
    evidence.input_identity(split)
    mean,scale=split['mean'],split['scale']
    if mean.ndim!=1 or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any(): raise ValueError('Invalid scaler')
    for part in ('fit','validation'):
        s=split[part];x=s['x'];y=s['y'];z=(x-mean)/scale
        if x.dtype!=np.float64 or x.ndim!=2 or x.shape[1]!=len(mean) or not np.isfinite(x).all(): raise ValueError('Invalid original features')
        if set(np.unique(y))!={0,1} or y.shape!=(len(x),): raise ValueError('Both binary classes required')
        for key in ('ids','groups'):
            if s[key].shape!=y.shape or s[key].dtype.kind not in 'iu': raise ValueError('Invalid row/group identity')
        if len(np.unique(s['ids']))!=len(y): raise ValueError('Duplicate row IDs')
        if not np.array_equal(z,s['standardized']) or not np.array_equal(.5+np.arctan(z)/np.pi,s['angles']): raise ValueError('Raw preprocessing mismatch')
    for key in ('ids','groups'):
        if np.intersect1d(split['fit'][key],split['validation'][key]).size: raise ValueError('Development overlap')
    if mode=='synthetic':
        if sum(len(split[p]['y']) for p in ('fit','validation'))>128 or any(len(str(split[k]))==64 for k in ('manifest_sha256','data_sha256','split_sha256')): raise ValueError('Synthetic input bound/identity violation')

def grid():
    p=train.protocol()['kernels']
    return dict(fidelity=[dict(C=c) for c in p['C']],projected=[dict(C=c,multiplier=g) for c in p['C'] for g in p['multipliers']])

def _identity(split,config,mode,kind,**extra):
    return dict(evidence.identity_for(split,config,mode,train.protocol()),kind=kind,**extra)

def save_inputs(out,split):
    save_npz(Path(out)/'preprocessing.npz',mean=split['mean'],scale=split['scale'])
    for p in ('fit','validation'):
        save_npz(Path(out)/(p+'_inputs.npz'),**{k:split[p][k] for k in ('x','y','ids','groups')})

def save_predictions(path,section,scores):
    save_npz(path,scores=np.asarray(scores,dtype=np.float64),**{k:section[k] for k in ('ids','groups','y')})

def _finish(out,identity,recipe,record):
    recipe=dict(recipe,source_hashes=identity['source_hashes'],refit=False,threshold=0.,native_pickle_required=False,original_feature_inference=True)
    data.write_json(Path(out)/'recipe.json',recipe)
    return evidence.seal(out,dict(record,config=identity['config'],kind=identity['kind']))

def _design(method,left,right,gamma):
    return model.fidelity_kernel(left,right) if method=='fidelity' else model.gaussian_kernel(left,right,gamma)

def fit(data_dir,out_dir,checkpoint,config,*,pin_path):
    """Fit both kernels for a new dataset checkpoint; literal formal grid only."""
    train._verify_pin(data_dir,pin_path)
    if set(config)!={'dataset','resample','init','arm'} or any(type(config.get(key)) is not int for key in ('resample','init')) or config['dataset']=='phishing' or config['arm'] not in ('initial','trained'): raise ValueError('Use authenticated Phishing import; invalid kernel coordinates')
    expected=next((c for c in train.make_tasks() if all(c[k]==config[k] for k in ('dataset','resample','init'))),None)
    if expected is None: raise ValueError('Unknown kernel task')
    completion=evidence.authenticate(checkpoint,expected_config=dict(expected,pin_sha256=data.digest(pin_path)))
    split=data.load_split(data_dir,config['dataset'],config['resample'])
    if read(Path(checkpoint)/'identity.json')['inputs']!=evidence.input_identity(split): raise ValueError('Checkpoint input mismatch')
    params=arrays(Path(checkpoint)/('initial.npz' if config['arm']=='initial' else 'best.npz'))
    refs=dict(completion_sha256=data.digest(Path(checkpoint)/'completion.json'),identity_sha256=data.digest(Path(checkpoint)/'identity.json'),pin_sha256=data.digest(pin_path))
    def verify_source():
        train._verify_pin(data_dir,pin_path)
        evidence.authenticate(checkpoint,expected_config=dict(expected,pin_sha256=data.digest(pin_path)))
        if data.digest(Path(checkpoint)/'completion.json')!=refs['completion_sha256'] or evidence.input_identity(data.load_split(data_dir,config['dataset'],config['resample']))!=evidence.input_identity(split):raise ValueError('Source/input changed during kernel fitting')
    return _fit(split,out_dir,params['theta'],params['probe'],config,grid(),mode='formal',references=refs,verify_source=verify_source)

def fit_synthetic(split,out_dir,theta,probe,config,*,candidates=None):
    if config.get('arm') not in ('initial','trained') or theta.shape[0]>4 or len(probe)>3: raise ValueError('Synthetic circuit bound')
    candidates=candidates or dict(fidelity=[dict(C=.1),dict(C=1.)],projected=[dict(C=.1,multiplier=.1),dict(C=1.,multiplier=1.)])
    if set(candidates)!=set(METHODS) or any(not 1<=len(v)<=2 for v in candidates.values()): raise ValueError('Synthetic candidate bound')
    return _fit(split,out_dir,theta,probe,config,candidates,mode='synthetic',references=None)

def _fit(split,out_dir,theta,probe,config,candidates,*,mode,references,verify_source=lambda:None):
    validate(split,mode);n=max(1,(len(split['mean'])-1).bit_length())
    if theta.dtype!=np.float64 or theta.ndim!=2 or theta.shape[1]!=15 or probe.dtype!=np.float64 or probe.shape!=(2**n-1,) or not np.isfinite(theta).all() or not np.isfinite(probe).all(): raise ValueError('Invalid circuit parameters')
    if mode=='formal' and theta.shape!=(9,15): raise ValueError('Formal T=8 required')
    identity=_identity(split,config,mode,'kernel',parameters={k:evidence._array_identity(v) for k,v in [('theta',theta),('probe',probe)]},references=references,candidates=candidates)
    out=Path(out_dir);existing=evidence.begin(out,identity)
    if existing is not None: return existing
    start,cpu=time.monotonic(),time.process_time();torch.set_num_threads(1)
    save_inputs(out,split);save_npz(out/'parameters.npz',theta=theta,probe=probe)
    selected={};outcomes=[];peak_design=0
    with threadpool_limits(limits=1):
        states={p:model.feature_states(split[p]['angles'],theta,probe,n=n) for p in ('fit','validation')}
        features={p:model.projected_features(states[p]) for p in states}
        save_predictions(out/'zz_validation.npz',split['validation'],features['validation'][:,-1])
        cache_bytes=sum(v.nbytes for v in states.values())+sum(v.nbytes for v in features.values())
        for method in METHODS:
            rep=states if method=='fidelity' else features;successful=[];models={};last=object();fit_design=val_design=None
            for index,candidate in enumerate(candidates[method]):
                current=dict(method=method,index=index,candidate=candidate);cs,cc=time.monotonic(),time.process_time()
                data.write_json(out/'progress.json',dict(stage='candidate',current=current,finished=len(outcomes)))
                try:
                    gamma=None if method=='fidelity' else model.projected_gamma(features['fit'],candidate['multiplier'])
                    if fit_design is None or gamma!=last:
                        fit_design=val_design=None
                        fit_design=_design(method,rep['fit'],rep['fit'],gamma);val_design=_design(method,rep['validation'],rep['fit'],gamma);last=gamma
                    peak_design=max(peak_design,fit_design.nbytes+val_design.nbytes)
                    estimator=SVC(C=candidate['C'],kernel='precomputed',class_weight='balanced',tol=.001,cache_size=512,max_iter=-1)
                    estimator.fit(fit_design,split['fit']['y']);scores=estimator.decision_function(val_design)
                    item=dict(current,status='complete',gamma=gamma,metrics=metrics(split['validation']['y'],scores),support_count=len(estimator.support_))
                    models[index]=dict(support_indices=estimator.support_.astype(np.int64),dual_coef=estimator.dual_coef_[0],intercept=estimator.intercept_,classes=estimator.classes_)
                    save_predictions(out/'candidates'/f'{method}_{index:03d}.npz',split['validation'],scores);successful.append(item)
                except Exception as exc:
                    item=dict(current,status='failed',error_type=type(exc).__name__,error=str(exc))
                item['costs']=costs(cs,cc);outcomes.append(item);data.write_json(out/'candidates'/f'{method}_{index:03d}.json',item)
            if successful:
                winner=min(successful,key=lambda v:selection_key(v,config['dataset']));selected[method]=winner
                compact=models[winner['index']];support=split['fit']['x'][compact['support_indices']]
                save_npz(out/'selected'/f'{method}.npz',**compact,support_x=support,support_ids=split['fit']['ids'][compact['support_indices']])
                shutil.copyfile(out/'candidates'/f"{method}_{winner['index']:03d}.npz",out/('validation_'+method+'.npz'))
            fit_design=val_design=None
    if any(v['status']!='complete' for v in outcomes): raise ValueError('Candidate failures retained; bundle unsealed')
    verify_source()
    if evidence.input_identity(split)!=identity['inputs']: raise ValueError('Input changed during fitting')
    resources=dict(num_data_qubits=n,num_processing_qubits=2,total_qubits=n+2,state_preparation_queries=len(theta)-1,
                   fidelity_pair_queries=2*(len(theta)-1),projected_feature_count=3*(n+2)+1,projected_measurement_bases_up_to=3,
                   equal_total_hardware_cost=False,cache_array_bytes=cache_bytes,peak_design_array_bytes=peak_design)
    return _finish(out,identity,dict(kind='kernel',model=dict(n=n,T=len(theta)-1),selected=selected,candidates=candidates),dict(selected=selected,candidate_count=len(outcomes),costs=costs(start,cpu),resources=resources))

def predict_original_features(out_dir,x,method,*,evidence_mode='formal'):
    out=Path(out_dir);evidence.authenticate(out,evidence_mode=evidence_mode);recipe=read(out/'recipe.json');pre=arrays(out/'preprocessing.npz')
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(pre['mean']) or not np.isfinite(x).all(): raise ValueError('Invalid raw feature input')
    if not len(x): return np.empty(0,dtype=np.float64)
    params=arrays(out/'parameters.npz')
    def states(raw): return model.feature_states(.5+np.arctan((raw-pre['mean'])/pre['scale'])/np.pi,params['theta'],params['probe'],n=recipe['model']['n'])
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        left=states(x)
        if recipe['kind']=='qrenn':
            if method!='qrenn_zz': raise ValueError('Unknown QRENN readout')
            return model.projected_features(left)[:,-1]
        if method not in METHODS: raise ValueError('Unknown kernel')
        selected=arrays(out/'selected'/f'{method}.npz');right=states(selected['support_x'])
        if method=='projected': left,right=model.projected_features(left),model.projected_features(right)
        return _design(method,left,right,recipe['selected'][method]['gamma'])@selected['dual_coef']+selected['intercept'][0]

def import_qrenn(data_dir,out_dir,checkpoint,config,*,pin_path=None,legacy_data=None,legacy_qrenn=None):
    """Convert an authenticated selected QRENN to a portable immutable bundle."""
    if set(config)!={'dataset','resample','init'} or any(type(config.get(key)) is not int for key in ('resample','init')) or config.get('resample') not in range(5) or config.get('init') not in range(3):raise ValueError('Literal QRENN import coordinates required')
    split=data.load_split(data_dir,config['dataset'],config['resample']);validate(split,'formal')
    if config['dataset']=='phishing':
        source=train.phishing_source(data_dir,legacy_data,legacy_qrenn,config['resample'],config['init'])
        checkpoint=Path(source['source']['path']);refs=source['source'];params=source['parameters']['best']
    else:
        train._verify_pin(data_dir,pin_path)
        expected=next(c for c in train.make_tasks() if all(c[k]==config[k] for k in ('dataset','resample','init')))
        completion=evidence.authenticate(checkpoint,expected_config=dict(expected,pin_sha256=data.digest(pin_path)))
        if read(Path(checkpoint)/'identity.json')['inputs']!=evidence.input_identity(split) or completion['actual_epochs']!=1000: raise ValueError('Invalid source QRENN')
        params=arrays(Path(checkpoint)/'best.npz');refs=dict(completion_sha256=data.digest(Path(checkpoint)/'completion.json'))
    return _import_qrenn(split,out_dir,checkpoint,config,params,refs,'formal')

def import_qrenn_synthetic(split,out_dir,checkpoint,config):
    evidence.authenticate(checkpoint,evidence_mode='synthetic')
    if read(Path(checkpoint)/'identity.json')['inputs']!=evidence.input_identity(split): raise ValueError('Input mismatch')
    return _import_qrenn(split,out_dir,checkpoint,config,arrays(Path(checkpoint)/'best.npz'),dict(completion_sha256=data.digest(Path(checkpoint)/'completion.json')),'synthetic')

def _import_qrenn(split,out_dir,checkpoint,config,params,refs,mode):
    validate(split,mode);out=Path(out_dir);identity=_identity(split,config,mode,'qrenn',references=refs)
    existing=evidence.begin(out,identity)
    if existing is not None:return existing
    save_inputs(out,split);save_npz(out/'parameters.npz',theta=params['theta'],probe=params['probe'])
    for name in ('initial.npz','best.npz','terminal.npz','history.json','validation.npz','completion.json','identity.json'):
        shutil.copyfile(Path(checkpoint)/name,out/('source_'+name))
    shutil.copyfile(Path(checkpoint)/'validation.npz',out/'validation_qrenn_zz.npz')
    return _finish(out,identity,dict(kind='qrenn',model=dict(n=(len(split['mean'])-1).bit_length(),T=params['theta'].shape[0]-1),selected={'qrenn_zz':dict(source='source_best.npz')}),dict(costs=read(Path(checkpoint)/'completion.json').get('costs',{})))


def import_phishing(data_dir,out_dir,kernel_source,config,*,legacy_data,legacy_qrenn):
    """Authenticate and copy a sealed prior kernel job; never fit a replacement."""
    from matched_kernel_study import evidence as old_evidence
    from matched_kernel_study.train import candidate_grid as old_grid
    if set(config)!={'dataset','resample','init','arm'} or any(type(config.get(key)) is not int for key in ('resample','init')) or config.get('dataset')!='phishing' or config.get('arm') not in ('initial','trained'): raise ValueError('Phishing source coordinates required')
    oldconfig={k:config[k] for k in ('arm','resample','init')};source=Path(kernel_source)
    completion=old_evidence.authenticate(source,expected_config=oldconfig)
    pin_path=source.parents[3]/'pin.json';oldpin=read(pin_path)
    if (oldpin.get('schema')!='qrenn-a07-matched-kernel-pin-v1' or oldpin['source_hashes']!=old_evidence.source_hashes()
            or oldpin['protocol']!=read(source/'protocol.json')
            or read(source/'identity.json')['references']['pin_sha256']!=data.digest(pin_path)):
        raise ValueError('Old kernel dependency pin mismatch')
    qsource=train.phishing_source(data_dir,legacy_data,legacy_qrenn,config['resample'],config['init'])
    identity_old=read(source/'identity.json');stage='initial' if config['arm']=='initial' else 'best'
    if identity_old['checkpoint']['completion_sha256']!=qsource['source']['completion_sha256']: raise ValueError('Kernel paired QRENN source mismatch')
    params=arrays(source/'parameters.npz')
    for k in ('theta','probe'):
        if not np.array_equal(params[k],qsource['parameters'][stage][k]): raise ValueError('Paired checkpoint parameters differ')
    split=data.load_split(data_dir,'phishing',config['resample']);validate(split,'formal')
    # Old manifest IDs differ; actual arrays/scaler must agree exactly (angles
    # receive only the previously authenticated two-epsilon allowance).
    pre=arrays(source/'preprocessing.npz');fit=arrays(source/'fit_inputs.npz')
    for k in ('mean','scale'):
        if not np.array_equal(pre[k],split[k]): raise ValueError('Old kernel scaler mismatch')
    for k in ('x','ids','groups','y'):
        if not np.array_equal(fit[k],split['fit'][k]): raise ValueError('Old kernel fitting rows differ')
    for p in ('fit','validation'):
        for k in ('ids','groups','y'):
            if not np.array_equal(pre[p+'_'+k],split[p][k]): raise ValueError('Old kernel partition identity differs')
    if identity_old['candidates']!=grid() or old_grid()!=grid() or completion['candidate_accounting']!=dict(planned=16,finished=16,successful=16,failed=0): raise ValueError('Old kernel grid/accounting differs')
    refs=dict(completion_sha256=data.digest(source/'completion.json'),identity_sha256=data.digest(source/'identity.json'),pin_sha256=data.digest(pin_path),paired_qrenn=qsource['source'])
    out=Path(out_dir);identity=_identity(split,config,'formal','kernel',references=refs,candidates=grid());existing=evidence.begin(out,identity)
    if existing is not None:return existing
    save_inputs(out,split);save_npz(out/'parameters.npz',theta=params['theta'],probe=params['probe'])
    shutil.copyfile(pin_path,out/'source_pin.json')
    for name in ('identity.json','completion.json','recipe.json','costs.json'):
        shutil.copyfile(source/name,out/('source_'+name))
    selected=read(source/'recipe.json')['selected']
    for method in METHODS:
        compact=arrays(source/'selected'/f'{method}.npz');indices=compact['support_indices']
        save_npz(out/'selected'/f'{method}.npz',**{k:compact[k] for k in ('support_indices','dual_coef','intercept','classes')},support_x=fit['x'][indices],support_ids=fit['ids'][indices])
        shutil.copyfile(source/'validation'/f'{method}.npz',out/('validation_'+method+'.npz'))
    shutil.copytree(source/'candidates',out/'candidates')
    old_evidence.authenticate(source,expected_config=oldconfig)
    if data.digest(pin_path)!=refs['pin_sha256']:raise ValueError('Old pin changed during import')
    return _finish(out,identity,dict(kind='kernel',model=dict(n=5,T=8),selected=selected,candidates=grid()),dict(selected=selected,candidate_count=16,resources=completion['resource'],costs=completion['costs'],reused_source=True))
