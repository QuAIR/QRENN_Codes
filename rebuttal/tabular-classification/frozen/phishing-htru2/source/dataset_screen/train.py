"""Joint universal-processing/probe optimization with immutable trajectory evidence."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_split, protocol, source_hashes, write_json
from .evaluate import (balanced_threshold, begin, identity_for, json_hash, metrics,
                       save_npz, seal)
from .model import initial_parameters, physical_probe, scores


def checkpoint_key(auc, weighted_loss, epoch):
    return (-float(auc), float(weighted_loss), int(epoch))


def plateau_decision(history, epoch, best_epoch):
    settings=protocol()['final']
    earlier=list(range(epoch+settings['plateau_earlier_epoch_offsets'][0],epoch+settings['plateau_earlier_epoch_offsets'][1]+1,5))
    recent=list(range(epoch+settings['plateau_recent_epoch_offsets'][0],epoch+1,5))
    losses={h['epoch']:h['train_loss'] for h in history}
    if len(earlier)!=10 or len(recent)!=10 or not all(e in losses for e in earlier+recent):
        raise ValueError('Incomplete literal plateau assessment windows')
    previous=float(np.mean([losses[e] for e in earlier])); current=float(np.mean([losses[e] for e in recent]))
    relative=(previous-current)/max(previous,1e-12)
    return dict(epoch=epoch,best_epoch=int(best_epoch),earlier_epochs=earlier,recent_epochs=recent,
                earlier_mean=previous,recent_mean=current,relative_improvement=relative,
                improvement_max=settings['plateau_relative_improvement_max'],best_epoch_age_min=settings['best_epoch_age_min'],
                stop=bool(relative<=settings['plateau_relative_improvement_max'] and best_epoch<=epoch-settings['best_epoch_age_min']),
                interpretation='Practical plateau rule, not proof of stationarity or a global optimum')


def validate_config(config):
    p=protocol(); c=config
    if c.get('dataset') not in p['datasets'] or type(c.get('resample')) is not int or c['resample'] not in range(5):
        raise ValueError('Invalid formal dataset/resample')
    if c.get('family') not in p['qrenn']['families'] or c.get('T') not in p['screening']['T'] or c.get('scale') not in p['screening']['scales']:
        raise ValueError('Invalid formal family/T/scale')
    if c.get('stage')=='screen':
        if c['resample']!=0 or c.get('init') not in range(2): raise ValueError('Invalid screening coordinate')
        seed=p['screening']['seeds'][c['init']]
    elif c.get('stage')=='final':
        if c.get('init') not in range(3) or not c.get('selection_sha256'): raise ValueError('Frozen screening choice required for final')
        seed=p['final']['seeds'][c['resample']][c['init']]
    else: raise ValueError('Formal stage must be screen or final')
    if c.get('seed')!=seed or c.get('shuffle_seed')!=seed+1000000: raise ValueError('Formal initialization/shuffle seed mismatch')
    if not c.get('manifest_sha256'): raise ValueError('Pinned manifest required')


def _loss(y, value, class_weights):
    return (class_weights[y]*(1-(2*y-1)*value)/2).mean()


def _parameter_artifact(path, theta, probe, features, family):
    save_npz(path,theta=theta.detach().cpu().numpy(),probe=probe.detach().cpu().numpy())
    write_json(Path(path).with_suffix('.physical.json'),physical_probe(probe.detach().cpu(),features,family))


def predict_parameters(parameters_path, config, angles):
    """Predict angle matrices using saved numeric parameters (no preprocessing fitting)."""
    with np.load(parameters_path,allow_pickle=False) as params:
        theta=torch.tensor(params['theta'],dtype=torch.float64); probe=torch.tensor(params['probe'],dtype=torch.float64)
    values=[]
    with torch.no_grad():
        for start in range(0,len(angles),512):
            values.append(scores(torch.tensor(angles[start:start+512],dtype=torch.float64),theta,probe,config['family'],config['scale']).numpy())
    return np.concatenate(values) if values else np.empty(0)


def fit(data_dir,out_dir,config):
    """Formal entrypoint; default loader never exposes test rows or labels."""
    c=dict(config); resume=bool(c.pop('resume',False)); validate_config(c)
    split=load_split(data_dir,c['dataset'],c['resample'])
    if split['manifest_sha256']!=c['manifest_sha256']: raise ValueError('Pinned manifest mismatch')
    split['raw_inputs']=json.loads((Path(data_dir)/'manifest.json').read_text())['datasets'][c['dataset']]
    if c.get('source_hashes_sha256',json_hash(source_hashes()))!=json_hash(source_hashes()): raise ValueError('Pinned sources differ')
    def verify_inputs():
        loaded=load_split(data_dir,c['dataset'],c['resample'])
        if any(loaded[k]!=split[k] for k in ('manifest_sha256','data_sha256','split_sha256')): raise ValueError('Pinned input changed during fitting')
    result=_fit(split,out_dir,c,evidence_mode='formal',epochs=None,batch_size=protocol()['training']['batch_size'],
                resume=resume,verify_inputs=verify_inputs)
    final=load_split(data_dir,c['dataset'],c['resample'])
    if any(final[k]!=split[k] for k in ('manifest_sha256','data_sha256','split_sha256')): raise ValueError('Input identity changed')
    return result


def fit_synthetic(split,out_dir,config,*,epochs=10,batch_size=7,resume=False):
    """Explicit bounded test API. Its completion can never satisfy a formal freeze."""
    if epochs<0 or epochs>30: raise ValueError('Synthetic budget must be 0..30 epochs')
    c=dict(config,evidence_mode='synthetic',synthetic_epochs=epochs,synthetic_batch_size=batch_size)
    return _fit(split,out_dir,c,evidence_mode='synthetic',epochs=epochs,batch_size=batch_size,resume=resume)


def _fit(split,out_dir,config,*,evidence_mode,epochs,batch_size,resume,verify_inputs=lambda:None):
    started=time.monotonic(); torch.set_num_threads(1); out=Path(out_dir)
    if set(split)&{'test'}: raise ValueError('Fitting input must exclude held-out test')
    identity=identity_for(split,config,evidence_mode)
    existing=begin(out,identity,resume=resume)
    if existing is not None: return existing
    device=torch.device(config.get('device','cpu')); p=protocol(); features=split['fit']['angles'].shape[1]
    theta,probe=initial_parameters(features,config['family'],config['T'],config['seed'])
    theta=torch.nn.Parameter(theta.to(device)); probe=torch.nn.Parameter(probe.to(device))
    optimizer=torch.optim.Adam([dict(params=[theta],lr=p['training']['theta_lr']),dict(params=[probe],lr=p['training']['probe_lr'])])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4,min_lr=[.002,.0002])
    shuffle=np.random.default_rng(config['shuffle_seed'])
    x={name:torch.tensor(split[name]['angles'],dtype=torch.float64,device=device) for name in ('fit','validation')}
    y={name:torch.tensor(split[name]['y'],dtype=torch.long,device=device) for name in ('fit','validation')}
    counts=np.bincount(split['fit']['y'],minlength=2)
    if not (counts>0).all(): raise ValueError('Both fitting classes required')
    weights=torch.tensor(len(x['fit'])/(2*counts),dtype=torch.float64,device=device)
    best_key=None; best_epoch=0; history=[]; decisions=[]; grad_history={}; current=0
    if resume and (out/'progress.json').exists():
        progress=json.loads((out/'progress.json').read_text())
        if digest(out/'checkpoint.pt')!=progress['checkpoint_sha256']: raise ValueError('Interrupted checkpoint mutated')
        state=torch.load(out/'checkpoint.pt',map_location=device,weights_only=False)
        if state['identity_sha256']!=digest(out/'identity.json'): raise ValueError('Resume identity mismatch')
        if not all(k in state for k in ('optimizer','scheduler','shuffle_rng','torch_rng','cuda_rng')): raise ValueError('Incomplete optimizer/RNG checkpoint')
        for name in ('best.npz','best.physical.json','history.json','initial.npz','initial.physical.json','preprocessing.npz'):
            if digest(out/name)!=progress['bound_artifacts'][name]: raise ValueError('Interrupted artifact mutated: '+name)
        with torch.no_grad(): theta.copy_(state['theta']); probe.copy_(state['probe'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        shuffle.bit_generator.state=state['shuffle_rng']; torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        current=state['epoch']; history=state['history']; best_key=tuple(state['best_key']); best_epoch=state['best_epoch']
        decisions=state['decisions']; grad_history=state['gradient_norms']
    else:
        if resume and any(path.name!='identity.json' for path in out.iterdir()): raise ValueError('No authenticated resumable checkpoint; attempt preserved')
        save_npz(out/'preprocessing.npz',mean=split['mean'],scale=split['scale'],fit_ids=split['fit']['ids'],
                 validation_ids=split['validation']['ids'],fit_groups=split['fit']['groups'],validation_groups=split['validation']['groups'],fit_y=split['fit']['y'],validation_y=split['validation']['y'])
        _parameter_artifact(out/'initial.npz',theta,probe,features,config['family'])
    cap=epochs if evidence_mode=='synthetic' else (p['screening']['epochs'] if config['stage']=='screen' else p['final']['max_epochs'])
    def predictions(name):
        blocks=[]
        with torch.no_grad():
            for start in range(0,len(x[name]),batch_size): blocks.append(scores(x[name][start:start+batch_size],theta,probe,config['family'],config['scale']))
        return torch.cat(blocks)
    def gradients():
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(x['fit']),batch_size):
            xb=x['fit'][start:start+batch_size]; yb=y['fit'][start:start+batch_size]
            (_loss(yb,scores(xb,theta,probe,config['family'],config['scale']),weights)*len(xb)/len(x['fit'])).backward()
        return dict(theta=float(torch.linalg.vector_norm(theta.grad)),probe=float(torch.linalg.vector_norm(probe.grad)))
    def assess(epoch):
        nonlocal best_key,best_epoch
        verify_inputs()
        if identity['source_hashes']!=source_hashes(): raise ValueError('Source changed during fitting')
        values={name:predictions(name) for name in ('fit','validation')}
        losses={name:float(_loss(y[name],values[name],weights)) for name in values}
        val=values['validation'].cpu().numpy(); threshold=balanced_threshold(split['validation']['y'],val)
        item=dict(epoch=epoch,train_loss=losses['fit'],validation_loss=losses['validation'],
                  train=metrics(split['fit']['y'],values['fit'].cpu().numpy(),threshold),
                  validation=metrics(split['validation']['y'],val,threshold),learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(item); key=checkpoint_key(item['validation']['roc_auc'],losses['validation'],epoch)
        if best_key is None or key<best_key:
            best_key=key; best_epoch=epoch; _parameter_artifact(out/'best.npz',theta,probe,features,config['family'])
        scheduler.step(losses['validation'])
        if epoch in p['final']['gradient_norm_epochs'] or epoch==cap: grad_history[str(epoch)]=gradients()
        if evidence_mode=='formal' and config['stage']=='final' and epoch in (1000,1500):
            decisions.append(plateau_decision(history,epoch,best_epoch))
        write_json(out/'history.json',history)
        state=dict(identity_sha256=digest(out/'identity.json'),epoch=epoch,theta=theta.detach(),probe=probe.detach(),
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),shuffle_rng=shuffle.bit_generator.state,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   history=history,best_key=best_key,best_epoch=best_epoch,decisions=decisions,gradient_norms=grad_history)
        temporary=out/'.checkpoint.pt'; torch.save(state,temporary); os.replace(temporary,out/'checkpoint.pt')
        bound={name:digest(out/name) for name in ('best.npz','best.physical.json','history.json','initial.npz','initial.physical.json','preprocessing.npz')}
        write_json(out/'progress.json',dict(epoch=epoch,best_epoch=best_epoch,checkpoint_sha256=digest(out/'checkpoint.pt'),bound_artifacts=bound))
    if not history: assess(0)
    stop=bool(decisions and decisions[-1]['stop'] and decisions[-1]['epoch']==current)
    while current<cap and not stop:
        current+=1
        order=shuffle.permutation(len(x['fit']))
        for offset in range(0,len(order),batch_size):
            indexes=order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            value=scores(x['fit'][indexes],theta,probe,config['family'],config['scale'])
            loss=_loss(y['fit'][indexes],value,weights)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite fitting loss')
            loss.backward(); optimizer.step()
        if current%5==0 or current==cap:
            assess(current)
            stop=bool(decisions and decisions[-1]['epoch']==current and decisions[-1]['stop'])
    terminal_grad=gradients(); _parameter_artifact(out/'terminal.npz',theta,probe,features,config['family'])
    selected=predict_parameters(out/'best.npz',config,split['validation']['angles'])
    threshold=balanced_threshold(split['validation']['y'],selected)
    selected_metrics=metrics(split['validation']['y'],selected,threshold)
    selected_loss=float(np.mean((len(x['fit'])/(2*counts))[split['validation']['y']]*(1-(2*split['validation']['y']-1)*selected)/2))
    # Reconstruction is intentionally separate from the incremental best-key update.
    reconstructed=min(history,key=lambda h:checkpoint_key(h['validation']['roc_auc'],h['validation_loss'],h['epoch']))
    if reconstructed['epoch']!=best_epoch or not np.isclose(selected_loss,reconstructed['validation_loss'],rtol=0,atol=1e-12):
        raise ValueError('Best checkpoint reconstruction failed')
    save_npz(out/'validation.npz',ids=split['validation']['ids'],y=split['validation']['y'],groups=split['validation']['groups'],scores=selected)
    write_json(out/'recipe.json',dict(schema='qrenn-a07-recipe-v1',kind='qrenn',config=config,source_hashes=identity['source_hashes'],
                                     preprocessing='preprocessing.npz',parameters='best.npz',threshold=threshold,
                                     score_convention='positive score favors label 1',original_feature_inference=True))
    verify_inputs()
    return seal(out,dict(kind='qrenn',config=config,best_epoch=best_epoch,actual_epochs=current,cap_hit=bool(current==cap),
                         plateau_decisions=decisions,gradient_norms=grad_history,terminal_gradient_norms=terminal_grad,
                         validation_metrics=selected_metrics,validation_loss=selected_loss,threshold=threshold,
                         elapsed_seconds=time.monotonic()-started,device=str(device),torch_threads=torch.get_num_threads()))
