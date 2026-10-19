"""Bounded staged campaign: complete screening, fresh finals, baselines, freeze, test."""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import baselines, train
from .data import digest, load_split, protocol, source_hashes, write_json
from .evaluate import (authenticate, balanced_threshold, json_hash, metrics,
                       predict_recipe, score_frozen)


def _same_write(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value: raise ValueError('Existing immutable selection differs: '+str(path))
    else: write_json(path,value)


def pin_inputs(data_dir,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); data_dir=Path(data_dir)
    pin=dict(schema='qrenn-a07-pin-v1',evidence_mode='formal',manifest_sha256=digest(data_dir/'manifest.json'),
             sources=source_hashes(),protocol_sha256=digest(Path(__file__).parent/'protocol.json'))
    _same_write(out/'pin.json',pin)
    for dataset,r in itertools.product(protocol()['datasets'],range(5)):
        split=load_split(data_dir,dataset,r)
        if split['manifest_sha256']!=pin['manifest_sha256']: raise ValueError('Manifest changed while pinning')
    return pin


def _pinned_config(config,pin):
    return dict(config,manifest_sha256=pin['manifest_sha256'],source_hashes_sha256=json_hash(pin['sources']))


def screen_configs():
    p=protocol(); items=[]
    for dataset,family,T,scale,init in itertools.product(p['datasets'],p['qrenn']['families'],p['screening']['T'],p['screening']['scales'],range(2)):
        seed=p['screening']['seeds'][init]
        items.append(dict(dataset=dataset,resample=0,family=family,T=T,scale=scale,init=init,stage='screen',seed=seed,shuffle_seed=seed+1000000))
    return items


def screen_path(config):
    c=config
    return f"screen/{c['dataset']}/{c['family']}/T{c['T']}/s{c['scale']:g}/i{c['init']}"


def _check_config(root,name,expected,mode):
    record=authenticate(root/name,evidence_mode=mode)
    identity=json.loads((root/name/'identity.json').read_text())
    if any(identity['config'].get(k)!=v for k,v in expected.items()): raise ValueError('Grid configuration mismatch: '+name)
    return record,identity


def select_screen(data_dir,out_dir,pin):
    out=Path(out_dir); p=protocol(); bound={}; selections={}
    for dataset in p['datasets']:
        split=load_split(data_dir,dataset,0); candidates=[]
        for family,T,scale in itertools.product(p['qrenn']['families'],p['screening']['T'],p['screening']['scales']):
            starts=[]
            for config in screen_configs():
                if (config['dataset'],config['family'],config['T'],config['scale'])!=(dataset,family,T,scale): continue
                name=screen_path(config); expected=_pinned_config(config,pin)
                record,identity=_check_config(out,name,expected,'formal')
                if record['actual_epochs']!=300: raise ValueError('Screening budget incomplete')
                _reconstruct(out/name,record,split)
                starts.append(dict(init=config['init'],validation_auc=record['validation_metrics']['roc_auc'],validation_loss=record['validation_loss'],
                                   completion_sha256=digest(out/name/'completion.json'),path=name))
                bound[name+'/completion.json']=digest(out/name/'completion.json')
            candidates.append(dict(family=family,T=T,scale=scale,starts=starts,
                                   mean_validation_roc_auc=float(np.mean([s['validation_auc'] for s in starts])),
                                   mean_validation_loss=float(np.mean([s['validation_loss'] for s in starts]))))
        # 16 coordinates with both starts, i.e. all 32 screening trajectories.
        chosen=min(candidates,key=lambda v:(-v['mean_validation_roc_auc'],v['mean_validation_loss'],v['T'],p['qrenn']['families'].index(v['family']),v['scale']))
        selection=dict(schema='qrenn-a07-screen-selection-v1',evidence_mode='formal',dataset=dataset,
                       manifest_sha256=pin['manifest_sha256'],pin_sha256=digest(out/'pin.json'),
                       candidates=candidates,selected={k:chosen[k] for k in ('family','T','scale')})
        path=out/'screen'/dataset/'selection.json'; _same_write(path,selection)
        bound[str(path.relative_to(out))]=digest(path); selections[dataset]=selection['selected']
    receipt=dict(schema='qrenn-a07-screening-completion-v1',evidence_mode='formal',pin_sha256=digest(out/'pin.json'),bound_files=bound,selections=selections)
    _same_write(out/'screening-completion.json',receipt)
    return selections


def screening_selections(out_dir,pin):
    out=Path(out_dir)
    try: receipt=json.loads((out/'screening-completion.json').read_text())
    except (OSError,ValueError) as exc: raise ValueError('Complete frozen screening selection required') from exc
    if receipt['evidence_mode']!='formal' or receipt['pin_sha256']!=digest(out/'pin.json'): raise ValueError('Screening identity mismatch')
    if pin['sources']!=source_hashes(): raise ValueError('Source changed after screening')
    expected={screen_path(c)+'/completion.json' for c in screen_configs()}|{f'screen/{d}/selection.json' for d in protocol()['datasets']}
    if set(receipt['bound_files'])!=expected: raise ValueError('Incomplete screening inventory')
    for name,sha in receipt['bound_files'].items():
        if digest(out/name)!=sha: raise ValueError('Screening selection/completion changed')
        if name.endswith('/completion.json'): authenticate((out/name).parent)
    return receipt['selections']


def expected_final(selections):
    result={}; p=protocol()
    for dataset in p['datasets']:
        for r in range(5):
            for init in range(3):
                seed=p['final']['seeds'][r][init]
                result[f'final/{dataset}/r{r}/i{init}']=dict(kind='qrenn',dataset=dataset,resample=r,init=init,
                    stage='final',seed=seed,shuffle_seed=seed+1000000,**selections[dataset])
            for method in baselines.CLASSICAL+baselines.QUANTUM:
                result[f'baselines/{dataset}/r{r}/{method}']=dict(kind='quantum' if method in baselines.QUANTUM else 'classical',
                                                                            dataset=dataset,resample=r,method=method)
    return result


def representative(records):
    return min(records,key=lambda r:(-r['validation_auc'],r['validation_loss'],r['init']))


def _reconstruct(path,record,split):
    """Recompute validation decisions from saved vectors and every candidate."""
    with np.load(path/'validation.npz',allow_pickle=False) as values:
        for key in ('ids','y','groups'):
            if not np.array_equal(values[key],split['validation'][key]): raise ValueError('Validation row identity mismatch')
        prediction=values['scores']
    threshold=balanced_threshold(split['validation']['y'],prediction)
    recalculated=metrics(split['validation']['y'],prediction,threshold)
    if recalculated!=record['validation_metrics'] or threshold!=record['threshold']: raise ValueError('Validation metrics/threshold mismatch')
    if record['kind']=='qrenn':
        history=json.loads((path/'history.json').read_text())
        chosen=min(history,key=lambda h:train.checkpoint_key(h['validation']['roc_auc'],h['validation_loss'],h['epoch']))
        if chosen['epoch']!=record['best_epoch']: raise ValueError('Checkpoint selection mismatch')
        counts=np.bincount(split['fit']['y'],minlength=2); weights=len(split['fit']['y'])/(2*counts)
        loss=float(np.mean(weights[split['validation']['y']]*(1-(2*split['validation']['y']-1)*prediction)/2))
        if not np.isclose(loss,record['validation_loss'],rtol=0,atol=1e-12): raise ValueError('Validation loss mismatch')
        for decision in record['plateau_decisions']:
            best_by_epoch=min((h for h in history if h['epoch']<=decision['epoch']),key=lambda h:train.checkpoint_key(h['validation']['roc_auc'],h['validation_loss'],h['epoch']))['epoch']
            if train.plateau_decision(history,decision['epoch'],best_by_epoch)!=decision: raise ValueError('Plateau reconstruction mismatch')
        if record['evidence_mode']=='formal' and record['config']['stage']=='final':
            actual=record['actual_epochs']
            if actual not in (1000,1500,2000): raise ValueError('Final budget incomplete')
            if actual<2000 and not any(d['epoch']==actual and d['stop'] for d in record['plateau_decisions']): raise ValueError('Unjustified early stop')
            if bool(record['cap_hit'])!=(actual==2000): raise ValueError('Cap marker mismatch')
    else:
        grid=json.loads((path/'candidate_grid.json').read_text())
        expected=(baselines.candidate_grids()[record['method']] if record['evidence_mode']=='formal' else record['config']['synthetic_candidates'])
        if grid['candidates']!=expected or record['candidate_count']!=len(expected): raise ValueError('Baseline grid incomplete')
        reconstructed=[]
        for index in range(len(expected)):
            with np.load(path/'candidates'/f'{index:03d}.npz',allow_pickle=False) as candidate:
                for key in ('ids','y','groups'):
                    if not np.array_equal(candidate[key],split['validation'][key]): raise ValueError('Candidate row identity mismatch')
                reconstructed.append((-float(roc_auc_score(candidate['y'],candidate['scores'])),index))
        chosen=min(reconstructed)[1]
        if record['selected_index']!=chosen: raise ValueError('Baseline candidate selection mismatch')
        with np.load(path/'candidates'/f'{chosen:03d}.npz',allow_pickle=False) as candidate:
            if not np.array_equal(prediction,candidate['scores']): raise ValueError('Selected vector differs from chosen candidate')


def _collect_models(out,pin,selections,splits,mode):
    expected=expected_final(selections); models={}; bound={}; records={}
    for name,coordinate in expected.items():
        config={k:v for k,v in coordinate.items() if k!='kind'}
        if coordinate['kind']=='qrenn' and mode=='formal': config['selection_sha256']=digest(out/'screen'/coordinate['dataset']/'selection.json')
        required=_pinned_config(config,pin) if mode=='formal' else config
        record,identity=_check_config(out,name,required,mode)
        if identity['manifest_sha256']!=pin['manifest_sha256'] or record['kind']!=coordinate['kind']: raise ValueError('Model identity mismatch')
        split=splits[coordinate['dataset'],coordinate['resample']]
        if any(identity[k]!=split[k] for k in ('data_sha256','split_sha256')): raise ValueError('Split/data identity mismatch')
        _reconstruct(out/name,record,split)
        records[name]=record
        models[name]=dict(coordinate,threshold=record['threshold'],completion_sha256=digest(out/name/'completion.json'))
        bound[name+'/completion.json']=digest(out/name/'completion.json')
    strong={}; representatives={}; gaps={}
    for dataset in protocol()['datasets']:
        strong[dataset]={}; representatives[dataset]={}; resample_gaps=[]
        for r in range(5):
            candidates=[f'baselines/{dataset}/r{r}/{m}' for m in baselines.QUANTUM]
            selected=min(candidates,key=lambda name:(-records[name]['validation_metrics']['roc_auc'],candidates.index(name)))
            strong[dataset][str(r)]=selected
            starts=[dict(init=i,validation_auc=records[f'final/{dataset}/r{r}/i{i}']['validation_metrics']['roc_auc'],
                         validation_loss=records[f'final/{dataset}/r{r}/i{i}']['validation_loss']) for i in range(3)]
            representatives[dataset][str(r)]=f"final/{dataset}/r{r}/i{representative(starts)['init']}"
            resample_gaps.append(float(np.mean([s['validation_auc'] for s in starts])-records[selected]['validation_metrics']['roc_auc']))
        gaps[dataset]=dict(resample_gaps=resample_gaps,mean_gap=float(np.mean(resample_gaps)))
    order=protocol()['evaluation']['dataset_tie_order']; recommendation=min(order,key=lambda dataset:(-gaps[dataset]['mean_gap'],order.index(dataset)))
    return dict(models=models,bound_files=bound,strong_quantum=strong,representative_qrenn=representatives,validation_gaps=gaps,recommended_dataset=recommendation)


def freeze(data_dir,out_dir):
    """Sequential freeze authenticating all64 screening,30 QRENN and60 baselines."""
    out=Path(out_dir); pin=pin_inputs(data_dir,out); selections=screening_selections(out,pin)
    # Re-run the literal screening selection from complete candidate evidence.
    if select_screen(data_dir,out,pin)!=selections: raise ValueError('Screening choice reconstruction differs')
    splits={(dataset,r):load_split(data_dir,dataset,r) for dataset,r in itertools.product(protocol()['datasets'],range(5))}
    inventory=_collect_models(out,pin,selections,splits,'formal')
    bound=inventory['bound_files']
    bound['screening-completion.json']=digest(out/'screening-completion.json')
    bound.update(json.loads((out/'screening-completion.json').read_text())['bound_files'])
    frozen=dict(schema='qrenn-a07-freeze-v1',evidence_mode='formal',pin_sha256=digest(out/'pin.json'),manifest_sha256=pin['manifest_sha256'],
                source_hashes=pin['sources'],bound_files=bound,models=inventory['models'],expected_prediction_files=90,
                strong_quantum=inventory['strong_quantum'],representative_qrenn=inventory['representative_qrenn'],all_qrenn_starts_retained=30,
                representative_rule='validation ROC-AUC descending, validation weighted loss ascending, init ascending',
                selected_screen=selections,recommended_dataset=inventory['recommended_dataset'],validation_gaps=inventory['validation_gaps'],
                recommendation_interpretation='Selection evidence only; not proof of matching',
                threshold_rule=protocol()['evaluation']['validation_balanced_threshold'])
    # Check pin after full reconstruction, before the atomic immutable receipt.
    if pin_inputs(data_dir,out)!=pin: raise ValueError('Identity changed during freeze')
    _same_write(out/'freeze.json',frozen)
    _same_write(out/'freeze-completion.json',dict(schema='qrenn-a07-freeze-completion-v1',freeze_sha256=digest(out/'freeze.json')))
    return frozen


def freeze_synthetic(out_dir,pin,selections,splits):
    """Explicit test-only complete 90-identity inventory; never valid for formal test."""
    out=Path(out_dir)
    if pin.get('evidence_mode')!='synthetic' or pin['sources']!=source_hashes(): raise ValueError('Synthetic pin/source mismatch')
    inventory=_collect_models(out,pin,selections,splits,'synthetic')
    result=dict(inventory,schema='qrenn-a07-freeze-v1',evidence_mode='synthetic',expected_prediction_files=90,
                all_qrenn_starts_retained=30,pin_sha256=digest(out/'pin.json'),manifest_sha256=pin['manifest_sha256'])
    _same_write(out/'freeze.json',result)
    _same_write(out/'freeze-completion.json',dict(freeze_sha256=digest(out/'freeze.json')))
    return result


def _worker(task):
    data_dir,out_dir,config,kind=task
    return (baselines.fit if kind=='baseline' else train.fit)(data_dir,out_dir,config)


def kernel_groups(tasks):
    """One sequential queue per device; faster workers cannot drift onto a busy GPU."""
    groups={}
    for task in tasks: groups.setdefault(task[2]['kernel_device'],[]).append(task)
    return list(groups.values())


def _kernel_worker(tasks):
    for task in tasks: _worker(task)
    return len(tasks)


def run_stage(stage,data_dir,out_dir,workers=1,kernel_workers=2,kernel_devices=('cpu',)):
    if workers<1 or kernel_workers<1 or not kernel_devices: raise ValueError('Positive worker limits and kernel devices required')
    if stage=='test': return score_frozen(data_dir,out_dir)
    if stage=='freeze': return freeze(data_dir,out_dir)
    pin=pin_inputs(data_dir,out_dir); out=Path(out_dir); tasks=[]
    if stage=='screen':
        tasks=[(str(data_dir),str(out/screen_path(c)),_pinned_config(c,pin),'qrenn') for c in screen_configs()]
    elif stage=='final':
        selections=screening_selections(out,pin)
        # Reconstruct before permitting any fresh final fit.
        select_screen(data_dir,out,pin)
        for name,coordinate in expected_final(selections).items():
            if coordinate['kind']!='qrenn': continue
            c={k:v for k,v in coordinate.items() if k!='kind'}
            c['selection_sha256']=digest(out/'screen'/c['dataset']/'selection.json')
            tasks.append((str(data_dir),str(out/name),_pinned_config(c,pin),'qrenn'))
    elif stage=='baselines':
        for dataset,r,method in itertools.product(protocol()['datasets'],range(5),baselines.CLASSICAL+baselines.QUANTUM):
            c=_pinned_config(dict(dataset=dataset,resample=r,method=method),pin)
            if method in baselines.QUANTUM: c['kernel_device']=kernel_devices[sum(t[2]['method'] in baselines.QUANTUM for t in tasks)%len(kernel_devices)]
            tasks.append((str(data_dir),str(out/f'baselines/{dataset}/r{r}/{method}'),c,'baseline'))
    else: raise ValueError('Unknown stage')
    # BLAS limits are inherited by spawn children before numeric imports.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    ordinary=[t for t in tasks if t[2].get('method') not in baselines.QUANTUM]
    kernels=[t for t in tasks if t[2].get('method') in baselines.QUANTUM]
    if ordinary:
        if workers==1:
            for task in ordinary: _worker(task)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for _ in pool.map(_worker,ordinary,chunksize=1): pass
    if kernels:
        groups=kernel_groups(kernels); count=min(kernel_workers,len(groups))
        if count==1:
            for group in groups: _kernel_worker(group)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
                for _ in pool.map(_kernel_worker,groups,chunksize=1): pass
    if stage=='screen': return select_screen(data_dir,out,pin)
    return dict(stage=stage,complete_tasks=len(tasks),manifest_sha256=pin['manifest_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['screen','final','baselines','freeze','test'])
    parser.add_argument('--data',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1); parser.add_argument('--kernel-workers',type=int,default=2)
    parser.add_argument('--kernel-devices',default='cpu')
    args=parser.parse_args()
    result=run_stage(args.stage,args.data,args.out,args.workers,args.kernel_workers,tuple(args.kernel_devices.split(',')))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
