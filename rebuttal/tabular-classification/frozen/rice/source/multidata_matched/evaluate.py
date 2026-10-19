"""Explicit all-model freeze, held-out evaluation, and independent scientific replay.

This module alone enables test loading. Evaluation never fits an estimator or
changes selection. NumPy/SciPy replay builds full states with a separate circuit
implementation and independent original Gell-Mann matrices.
"""
from __future__ import annotations
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from scipy.stats import rankdata
from dataset_screen.evaluate import json_hash
from . import data,evidence,train,kernels as k,baselines


def _methods(recipe):return list(recipe['selected'])

def _coordinates(identity):
    c=identity['config'];return (identity['kind'],c['dataset'],c['resample'],c.get('init'),c.get('arm'))

def expected_coordinates():
    p=train.protocol();result=set()
    for d in p['datasets']:
        for r in range(p['resamples']):
            result.add(('classical',d,r,None,None))
            for i in range(p['initializations']):
                result.add(('qrenn',d,r,i,None))
                for arm in p['kernels']['arms']:result.add(('kernel',d,r,i,arm))
    return result

def check_formal_inventory(records,selected_count):
    """Cross-check literal counts against the exact coordinate/method inventory."""
    p=train.protocol();expected=expected_coordinates()
    methods={'qrenn':{'qrenn_zz'},'kernel':set(p['kernels']['methods']),
             'classical':set(p['classical']['methods'])}
    coordinates=[tuple(r['coordinate']) for r in records]
    if len(set(coordinates))!=len(coordinates) or set(coordinates)!=expected:
        raise ValueError('Incomplete or duplicate formal bundle inventory')
    bundles={kind:sum(c[0]==kind for c in expected) for kind in methods}
    counts={kind:bundles[kind]*len(methods[kind]) for kind in methods}
    counts['total']=sum(counts.values())
    if bundles!=p['evaluation']['bundle_counts'] or counts!=p['evaluation']['selected_counts']:
        raise ValueError('Literal declared counts differ from formal coordinate inventory')
    if selected_count!=counts['total'] or any(len(r['methods'])!=len(methods[r['coordinate'][0]]) or set(r['methods'])!=methods[r['coordinate'][0]] for r in records):
        raise ValueError('Incomplete formal selected method inventory')


def check_candidates(out,*,mode='formal'):
    """Reconstruct every candidate metric and winner from saved predictions."""
    out=Path(out);evidence.authenticate(out,evidence_mode=mode);identity=k.read(out/'identity.json');recipe=k.read(out/'recipe.json');section=k.arrays(out/'validation_inputs.npz');checks=[]
    pre=k.arrays(out/'preprocessing.npz')
    for key in ('mean','scale'):
        if evidence._array_identity(pre[key])!=identity['inputs'][key]:raise ValueError('Stored scaler identity differs')
    for part in ('fit','validation'):
        stored=k.arrays(out/(part+'_inputs.npz'))
        for key in ('x','ids','groups','y'):
            if evidence._array_identity(stored[key])!=identity['inputs'][part][key]:raise ValueError('Stored raw input identity differs')
    if recipe['kind']=='qrenn':
        history=k.read(out/'source_history.json');completion=k.read(out/'source_completion.json');primary=train.primary_metric(identity['config']['dataset'])
        best=min(history,key=lambda h:train.checkpoint_key(h['validation'][primary],h['validation_loss'],h['epoch']))
        if best['epoch']!=completion['best_epoch']:raise ValueError('QRENN best epoch selection differs')
        for key in ('theta','probe'):
            if not np.array_equal(k.arrays(out/'parameters.npz')[key],k.arrays(out/'source_best.npz')[key]):raise ValueError('QRENN selected parameters differ')
        predictions=k.arrays(out/'validation_qrenn_zz.npz')
        computed=k.metrics(predictions['y'],predictions['scores'])
        for key,value in computed.items():
            if abs(value-best['validation'][key])>1e-12:raise ValueError('QRENN best validation metrics differ')
        return dict(kind='qrenn',best_epoch=best['epoch'],history_count=len(history))
    for method,candidates in recipe['candidates'].items():
        records=[]
        for index,candidate in enumerate(candidates):
            item=k.read(out/'candidates'/f'{method}_{index:03d}.json');pred=k.arrays(out/'candidates'/f'{method}_{index:03d}.npz')
            if item['status']!='complete' or item['index']!=index or item['candidate']!=candidate:raise ValueError('Candidate record differs')
            for key in ('ids','groups','y'):
                if not np.array_equal(pred[key],section[key]):raise ValueError('Candidate prediction rows differ')
            computed=k.metrics(pred['y'],pred['scores'])
            if computed!=item['metrics']:raise ValueError('Candidate metrics differ from predictions')
            records.append(item)
        winner=min(records,key=lambda v:k.selection_key(v,identity['config']['dataset']))
        if recipe['selected'][method]!=winner:raise ValueError('Selected candidate is not the declared winner')
        selected=k.arrays(out/('validation_'+method+'.npz'));candidate=k.arrays(out/'candidates'/f"{method}_{winner['index']:03d}.npz")
        if selected.keys()!=candidate.keys() or any(not np.array_equal(selected[key],candidate[key]) for key in candidate):raise ValueError('Selected validation prediction differs')
        compact=k.arrays(out/'selected'/f'{method}.npz')
        if recipe['kind']=='kernel' or method=='rbf_svm':
            fitting=k.arrays(out/'fit_inputs.npz');indices=compact['support_indices']
            if indices.dtype.kind not in 'iu' or indices.ndim!=1 or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(fitting['x'])):raise ValueError('Invalid original fitting support indices')
            if not np.array_equal(compact['support_x'],fitting['x'][indices]):raise ValueError('Support rows differ from original fitting rows')
            if recipe['kind']=='kernel' and not np.array_equal(compact['support_ids'],fitting['ids'][indices]):raise ValueError('Support row IDs differ')
            if compact['dual_coef'].shape!=indices.shape or compact['intercept'].shape!=(1,):raise ValueError('Invalid portable SVM coefficients')
            if winner['support_count']!=len(indices):raise ValueError('Support count differs')
        checks.append(dict(method=method,candidates=len(records),selected_index=winner['index']))
    return dict(kind=recipe['kind'],checks=checks)

def freeze(data_dir,bundles,snapshot,*,pin_path):
    try:train._verify_pin(data_dir,pin_path)
    except OSError as exc:raise ValueError('Missing source pin') from exc
    return _freeze(data_dir,bundles,snapshot,'formal')

def freeze_synthetic(bundles,snapshot):
    """Fixture-only freeze with <=6 bundles; cannot authorize formal evaluation."""
    if not 1<=len(bundles)<=6:raise ValueError('Synthetic freeze bound')
    return _freeze(None,bundles,snapshot,'synthetic')

def _freeze(data_dir,bundles,snapshot,mode):
    target=Path(snapshot)
    if target.exists():raise ValueError('Frozen snapshot immutable')
    records=[];coordinates=set();selected_count=0
    for location in bundles:
        out=Path(location).resolve();completion=evidence.authenticate(out,evidence_mode=mode);identity=k.read(out/'identity.json');recipe=k.read(out/'recipe.json');coord=_coordinates(identity)
        if coord in coordinates:raise ValueError('Duplicate bundle coordinate')
        coordinates.add(coord)
        expected={'qrenn':{'qrenn_zz'},'kernel':set(k.METHODS),'classical':set(baselines.METHODS)}[identity['kind']]
        if set(_methods(recipe))!=expected:raise ValueError('Incomplete selected method bundle')
        if mode=='formal':
            split=data.load_split(data_dir,identity['config']['dataset'],identity['config']['resample'])
            if evidence.input_identity(split)!=identity['inputs']:raise ValueError('Bundle source/data identity differs')
            if identity['kind']!='qrenn':
                planned=k.grid() if identity['kind']=='kernel' else baselines.grid(len(split['mean']))
                if recipe['candidates']!=planned:raise ValueError('Formal candidate grid changed')
        audit=check_candidates(out,mode=mode);selected_count+=len(expected)
        records.append(dict(path=str(out),completion_sha256=data.digest(out/'completion.json'),identity_sha256=data.digest(out/'identity.json'),coordinate=list(coord),methods=sorted(expected),selection_audit=audit))
    if mode=='formal':check_formal_inventory(records,selected_count)
    result=dict(schema='qrenn-a07-multidata-freeze-v1',evidence_mode=mode,source_hashes=evidence.source_hashes(),protocol=train.protocol(),
                manifest_sha256=None if mode=='synthetic' else data.digest(Path(data_dir)/'manifest.json'),bundles=records,selected_count=selected_count,
                interpretation=train.protocol()['reporting']['interpretation'])
    result['snapshot_sha256']=json_hash(result);data.write_json(target,result);return result

def authenticate_snapshot(snapshot,*,mode='formal'):
    value=k.read(snapshot);body={key:v for key,v in value.items() if key!='snapshot_sha256'}
    if value.get('schema')!='qrenn-a07-multidata-freeze-v1' or value.get('evidence_mode')!=mode or value['snapshot_sha256']!=json_hash(body):raise ValueError('Invalid frozen snapshot')
    if value['source_hashes']!=evidence.source_hashes() or value['protocol']!=train.protocol():raise ValueError('Frozen source/protocol changed')
    if len({tuple(r['coordinate']) for r in value['bundles']})!=len(value['bundles']):raise ValueError('Duplicate frozen coordinates')
    if sum(len(r['methods']) for r in value['bundles'])!=value['selected_count']:raise ValueError('Frozen selected count differs')
    for record in value['bundles']:
        out=Path(record['path']);evidence.authenticate(out,evidence_mode=mode)
        if sorted(_methods(k.read(out/'recipe.json')))!=record['methods']:raise ValueError('Frozen selected methods differ')
        if data.digest(out/'completion.json')!=record['completion_sha256'] or data.digest(out/'identity.json')!=record['identity_sha256'] or list(_coordinates(k.read(out/'identity.json')))!=record['coordinate']:raise ValueError('Frozen model identity changed')
    if mode=='formal':check_formal_inventory(value['bundles'],value['selected_count'])
    return value

def relocate_snapshot(snapshot,bundle_root,new_snapshot):
    """Explicit relocation preserving every frozen byte identity and coordinate.

    bundle_root contains a directory per original bundle basename. Ambiguous
    names are rejected. The relocated envelope has its own digest and records
    the parent envelope digest; no model/source identity changes are allowed.
    """
    original=k.read(snapshot)
    if original['snapshot_sha256']!=json_hash({a:b for a,b in original.items() if a!='snapshot_sha256'}):raise ValueError('Invalid parent freeze')
    names=[Path(r['path']).name for r in original['bundles']]
    if len(set(names))!=len(names):raise ValueError('Relocation requires unique bundle basenames')
    changed=dict(original,parent_snapshot_sha256=original['snapshot_sha256'])
    changed['bundles']=[dict(r,path=str((Path(bundle_root)/Path(r['path']).name).resolve())) for r in original['bundles']]
    changed['snapshot_sha256']=json_hash({a:b for a,b in changed.items() if a!='snapshot_sha256'})
    if Path(new_snapshot).exists():raise ValueError('Existing relocation preserved')
    data.write_json(new_snapshot,changed)
    try:authenticate_snapshot(new_snapshot,mode=original['evidence_mode'])
    except Exception:
        Path(new_snapshot).unlink();raise
    return changed

def numpy_states(x,mean,scale,theta,probe):
    """Independent full-state SciPy implementation; no production state calls."""
    n=max(1,(x.shape[1]-1).bit_length());dim=2**n;generators=[]
    for i in range(4):
        for j in range(4):
            if i==j==0:continue
            a=np.zeros((4,4),dtype=np.complex128)
            if i==j:
                factor=np.sqrt(2/(i*(i+1)));a[np.arange(i),np.arange(i)]=factor;a[i,i]=-i*factor
            elif i<j:a[i,j]=a[j,i]=1
            else:a[i,j]=1j;a[j,i]=-1j
            generators.append(a)
    blocks=[expm(1j*np.tensordot(row,np.asarray(generators),axes=1)) for row in theta]
    logits=np.r_[0.,probe];weights=np.exp(logits-logits.max());psi=np.sqrt(weights/weights.sum())
    angles=.5+np.arctan((x-mean)/scale)/np.pi;phases=np.exp(2j*np.pi*np.pad(angles,((0,0),(0,dim-x.shape[1]))))
    result=np.zeros((len(x),4,dim),dtype=np.complex128);result[:,0,:]=psi
    for layer,block in enumerate(blocks):
        # Explicit independent branch multiplication, rather than cached moments.
        result=np.stack([sum(block[a,b]*result[:,b,:] for b in range(4)) for a in range(4)],axis=1)
        if layer<len(blocks)-1:result[:,3,:]*=phases if layer%2==0 else phases.conj()
    return result.reshape(len(x),4*dim)

def numpy_pauli(states):
    q=states.shape[1].bit_length()-1;index=np.arange(states.shape[1]);columns=[]
    for bit in range(q):
        mask=1<<(q-bit-1);zero=index[(index&mask)==0];one=zero|mask
        overlap=np.sum(states[:,zero].conj()*states[:,one],axis=1)
        columns.extend([2*overlap.real,2*overlap.imag,np.sum(abs(states[:,zero])**2-abs(states[:,one])**2,axis=1)])
    sign=(1-2*((index>>(q-1))&1))*(1-2*((index>>(q-2))&1))
    columns.append(abs(states)**2@sign)
    return np.stack(columns,axis=1)

def independent_predict(out,x,method):
    out=Path(out);recipe=k.read(out/'recipe.json');pre=k.arrays(out/'preprocessing.npz')
    if recipe['kind']=='classical':return baselines.predict_original_features(out,x,method,evidence_mode=k.read(out/'identity.json')['evidence_mode'])
    params=k.arrays(out/'parameters.npz');left=numpy_states(x,pre['mean'],pre['scale'],params['theta'],params['probe'])
    if recipe['kind']=='qrenn':return numpy_pauli(left)[:,-1]
    selected=k.arrays(out/'selected'/f'{method}.npz');right=numpy_states(selected['support_x'],pre['mean'],pre['scale'],params['theta'],params['probe'])
    if method=='fidelity':design=abs(left.conj()@right.T)**2
    else:
        a,b=numpy_pauli(left),numpy_pauli(right);dist=sum((a[:,j,None]-b[None,:,j])**2 for j in range(a.shape[1]))
        design=np.exp(-recipe['selected'][method]['gamma']*dist)
    return design@selected['dual_coef']+selected['intercept'][0]

def discrepancy(saved,replayed):
    saved=np.asarray(saved);replayed=np.asarray(replayed)
    if saved.shape!=replayed.shape or not np.isfinite(replayed).all():raise ValueError('Invalid independent replay')
    return dict(max_abs_score_error=float(np.max(abs(saved-replayed),initial=0.)),classification_disagreements=int(np.count_nonzero((saved>=0)!=(replayed>=0))),
                ranking_disagreements=int(np.count_nonzero(rankdata(saved,method='average')!=rankdata(replayed,method='average'))),score_atol=train.protocol()['evaluation']['score_atol'],predictions_rounded=False)

def _name(record,method):
    kind,d,r,i,arm=record['coordinate'];return f'{d}_r{r}_'+('classical' if kind=='classical' else f'i{i}_{kind}'+('' if arm is None else '_'+arm))+'_'+method

def _method(record,method):
    if record['coordinate'][0]=='kernel':return record['coordinate'][4]+'_'+method+'_svm'
    return method

def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate(rows):
    """Mean starts within resample, then mean/sample SD across resamples."""
    grouped={}
    for row in rows:
        key=(row['partition'],row['dataset'],row['method'],int(row['resample']));grouped.setdefault(key,[]).append(row)
    resamples=[]
    for (partition,d,method,r),items in sorted(grouped.items()):
        resamples.append(dict(partition=partition,dataset=d,method=method,resample=r,starts=len(items),**{metric:float(np.mean([v[metric] for v in items])) for metric in train.protocol()['evaluation']['metrics']}))
    summary=[]
    for key in sorted({(r['partition'],r['dataset'],r['method']) for r in resamples}):
        items=[r for r in resamples if (r['partition'],r['dataset'],r['method'])==key];item=dict(partition=key[0],dataset=key[1],method=key[2],resamples=len(items))
        for metric in train.protocol()['evaluation']['metrics']:
            values=[r[metric] for r in items];item[metric+'_mean']=float(np.mean(values));item[metric+'_sample_sd']=None if len(values)<2 else float(np.std(values,ddof=1))
        summary.append(item)
    differences=[]
    for partition,d in sorted({(r['partition'],r['dataset']) for r in resamples}):
        methods=sorted({r['method'] for r in resamples if (r['partition'],r['dataset'])==(partition,d)})
        lookup={(r['method'],r['resample']):r for r in resamples if (r['partition'],r['dataset'])==(partition,d)}
        for left,right in itertools.combinations(methods,2):
            shared=sorted({r for m,r in lookup if m==left}&{r for m,r in lookup if m==right})
            for r in shared:
                differences.append(dict(partition=partition,dataset=d,left=left,right=right,resample=r,**{metric:lookup[left,r][metric]-lookup[right,r][metric] for metric in train.protocol()['evaluation']['metrics']}))
    return dict(resample_means=resamples,summary=summary,paired_differences=differences,interpretation=train.protocol()['reporting']['interpretation'])

def evaluate(data_dir,snapshot,out_dir):
    frozen=authenticate_snapshot(snapshot)
    if data.digest(Path(data_dir)/'manifest.json')!=frozen['manifest_sha256']:raise ValueError('Frozen data manifest differs')
    return _evaluate(frozen,out_dir,lambda c:data.load_split(data_dir,c[1],c[2],include_test=True),'formal')

def evaluate_synthetic(snapshot,out_dir,splits):
    frozen=authenticate_snapshot(snapshot,mode='synthetic')
    return _evaluate(frozen,out_dir,lambda c:splits[(c[1],c[2])],'synthetic')

def _evaluate(frozen,out_dir,loader,mode):
    out=Path(out_dir)
    if out.exists():raise ValueError('Evaluation attempt immutable; preserve failed attempt and choose a new destination')
    out.mkdir(parents=True);data.write_json(out/'freeze.json',frozen);rows=[];audits=[];evaluations=[]
    for record in frozen['bundles']:
        location=Path(record['path']);identity=k.read(location/'identity.json');split=loader(record['coordinate']);development={a:b for a,b in split.items() if a!='test'}
        if isinstance(development.get('groups'),dict):development['groups']={a:b for a,b in development['groups'].items() if a!='test'}
        if evidence.input_identity(development)!=identity['inputs']:raise ValueError('Evaluation input differs from frozen development data')
        if 'test' not in split:raise ValueError('Explicit held-out section required')
        for key in ('ids','groups'):
            if any(np.intersect1d(split['test'][key],split[p][key]).size for p in ('fit','validation')):raise ValueError('Held-out rows/groups overlap development')
        for method in record['methods']:
            started=time.monotonic();predict=baselines.predict_original_features if identity['kind']=='classical' else k.predict_original_features
            for partition in ('validation','test'):
                section=split[partition];scores=predict(location,section['x'],method,evidence_mode=mode);name=_name(record,method)
                if partition=='validation':
                    saved=k.arrays(location/('validation_'+method+'.npz'))
                    for key in ('ids','groups','y'):
                        if not np.array_equal(saved[key],section[key]):raise ValueError('Frozen validation rows differ')
                    replay=discrepancy(saved['scores'],scores)
                    if replay['max_abs_score_error']>train.protocol()['evaluation']['score_atol']:raise ValueError('Frozen validation score replay differs')
                    # All table values use the already selected saved predictions.
                    scores=saved['scores']
                k.save_predictions(out/partition/(name+'.npz'),section,scores)
                independent=independent_predict(location,section['x'],method);audit=discrepancy(scores,independent)
                if audit['max_abs_score_error']>train.protocol()['evaluation']['score_atol']:raise ValueError('Independent state replay failed')
                audits.append(dict(name=name,partition=partition,**audit))
                prediction=k.arrays(out/partition/(name+'.npz'))
                rows.append(dict(partition=partition,dataset=record['coordinate'][1],resample=record['coordinate'][2],init=record['coordinate'][3],method=_method(record,method),prediction_path=partition+'/'+name+'.npz',prediction_sha256=data.digest(out/partition/(name+'.npz')),**k.metrics(prediction['y'],prediction['scores'])))
            evaluations.append(dict(name=name,elapsed_seconds=time.monotonic()-started,completion_sha256=record['completion_sha256'],support_count=k.read(location/'recipe.json')['selected'][method].get('support_count'),validation_rows=len(split['validation']['y']),test_rows=len(split['test']['y'])))
        data.write_json(out/'progress.json',dict(completed_selected_models=len(evaluations),planned=frozen['selected_count']))
    result=aggregate(rows);result.update(selected_model_count=len(evaluations),model_costs=[dict(coordinate=r['coordinate'],costs=k.read(Path(r['path'])/'completion.json').get('costs',{}),resources=k.read(Path(r['path'])/'completion.json').get('resources',{})) for r in frozen['bundles']],evaluation_costs=evaluations)
    _csv(out/'runs.csv',rows);_csv(out/'resample_means.csv',result['resample_means']);_csv(out/'table.csv',result['summary']);_csv(out/'paired_differences.csv',result['paired_differences']);data.write_json(out/'summary.json',result);data.write_json(out/'replay_audit.json',audits)
    # Verify immutable model inventories again, after every evaluation finished.
    for record in frozen['bundles']:
        evidence.authenticate(record['path'],evidence_mode=mode)
        if data.digest(Path(record['path'])/'completion.json')!=record['completion_sha256']:raise ValueError('Model changed during evaluation')
    completion=dict(schema='qrenn-a07-multidata-evaluation-v1',evidence_mode=mode,snapshot_sha256=frozen['snapshot_sha256'],selected_model_count=len(evaluations),artifacts=evidence.inventory(out,exclude=('completion.json',)))
    data.write_json(out/'independent_audit.json',_audit_evaluation(out,completion))
    completion['artifacts']=evidence.inventory(out,exclude=('completion.json',))
    data.write_json(out/'completion.json',completion)
    return result


def audit_evaluation(out_dir):
    """Recompute saved metrics/selections and independently check hierarchy arithmetic."""
    out=Path(out_dir)
    return _audit_evaluation(out,k.read(out/'completion.json'))

def _audit_evaluation(out,completion):
    if completion['artifacts']!=evidence.inventory(out,exclude=('completion.json',)):raise ValueError('Evaluation artifact inventory changed')
    frozen=authenticate_snapshot(out/'freeze.json',mode=completion['evidence_mode'])
    for record in frozen['bundles']:check_candidates(record['path'],mode=completion['evidence_mode'])
    with (out/'runs.csv').open(newline='') as file:rows=list(csv.DictReader(file))
    if len(rows)!=2*frozen['selected_count']:raise ValueError('Evaluation prediction count differs')
    for row in rows:
        path=out/row['prediction_path']
        if data.digest(path)!=row['prediction_sha256']:raise ValueError('Prediction checksum differs')
        pred=k.arrays(path);computed=k.metrics(pred['y'],pred['scores'])
        for metric,value in computed.items():
            if value!=float(row[metric]):raise ValueError('Run metric differs from prediction')
    summary=k.read(out/'summary.json');checked=0
    for item in summary['summary']:
        selected=[r for r in rows if all(r[key]==item[key] for key in ('partition','dataset','method'))]
        resamples=sorted({r['resample'] for r in selected})
        if len(resamples)!=item['resamples']:raise ValueError('Hierarchy resample count differs')
        if completion['evidence_mode']=='formal':
            expected_starts=1 if item['method'] in baselines.METHODS else train.protocol()['initializations']
            if len(resamples)!=train.protocol()['resamples'] or any(sum(r['resample']==resample for r in selected)!=expected_starts for resample in resamples):raise ValueError('Formal hierarchy count differs')
        for metric in train.protocol()['evaluation']['metrics']:
            means=[]
            for resample in resamples:
                values=[float(r[metric]) for r in selected if r['resample']==resample];means.append(sum(values)/len(values))
            mean=sum(means)/len(means);sd=None if len(means)==1 else (sum((value-mean)**2 for value in means)/(len(means)-1))**.5
            if abs(mean-item[metric+'_mean'])>1e-14 or (sd is None)!=(item[metric+'_sample_sd'] is None) or sd is not None and abs(sd-item[metric+'_sample_sd'])>1e-14:raise ValueError('Independent hierarchy arithmetic differs')
            checked+=1
    return dict(predictions=len(rows),selected_models=frozen['selected_count'],hierarchy_metric_checks=checked,all_candidate_selections_checked=True)
