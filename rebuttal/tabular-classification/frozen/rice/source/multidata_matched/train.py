"""Fixed-budget paired training, strict evidence and original-feature replay.

Formal CLI and bounded synthetic API are deliberately separate. No call in this
module enables held-out loading. Assessment/update ordering follows the retained
original runner, including the scheduler assessment at epoch zero.
"""
from __future__ import annotations

import argparse
import json
import numbers
import os
import platform
import resource
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_split, write_json
from dataset_screen.evaluate import metrics, save_npz
from dataset_screen.model import initial_parameters, physical_probe
from . import evidence
from eigenbasis_study import model

PROTOCOL_PATH=Path(__file__).with_name('protocol.json')


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def checkpoint_key(auc,weighted_loss,epoch):
    return (-float(auc),float(weighted_loss),int(epoch))


def make_tasks():
    p=protocol()
    return [dict(arm='fixed',resample=r,init=i,dataset=d,n=p['registers'][d],T=p['model']['T'],scale=p['model']['scale'],
                 seed=p['seed_base']+p['seed_resample_stride']*r+i,
                 shuffle_seed=p['seed_base']+p['seed_resample_stride']*r+i+p['shuffle_seed_offset'],device='cpu')
            for d in p['datasets'] for r in range(p['resamples']) for i in range(p['initializations'])]


def primary_metric(dataset):
    return protocol()['primary_metrics'].get(dataset,'roc_auc')


def create_pin(data_dir,path):
    """Freeze source/protocol and manifest bytes before launching any formal job."""
    target=Path(path)
    if target.exists(): raise ValueError('Existing source pin preserved')
    value=dict(schema='qrenn-a07-multidata-pin-v1',source_hashes=evidence.source_hashes(),
               protocol_sha256=digest(PROTOCOL_PATH),manifest_sha256=digest(Path(data_dir)/'manifest.json'))
    write_json(target,value)
    return value


def _verify_pin(data_dir,pin_path):
    pin=json.loads(Path(pin_path).read_text())
    if pin.get('schema')!='qrenn-a07-multidata-pin-v1': raise ValueError('Invalid source pin schema')
    if pin['source_hashes']!=evidence.source_hashes(): raise ValueError('Pinned scientific source mismatch')
    if pin['protocol_sha256']!=digest(PROTOCOL_PATH): raise ValueError('Pinned protocol mismatch')
    if pin['manifest_sha256']!=digest(Path(data_dir)/'manifest.json'): raise ValueError('Pinned manifest mismatch')
    protocol()
    return pin


def fit(data_dir,out_dir,config,*,pin_path,resume=False):
    """Formal entrypoint: literal 1000 epochs, authenticated fit/validation only."""
    _validate_scale(config.get('scale'))
    if (config not in make_tasks() or any(type(config.get(k)) is not int for k in
            ('resample','init','n','T','seed','shuffle_seed'))):
        raise ValueError('Formal configuration must equal one of the declared training tasks')
    pin=_verify_pin(data_dir,pin_path)
    pin_digest=digest(pin_path)
    split=load_split(data_dir,config['dataset'],config['resample'])
    if split['manifest_sha256']!=pin['manifest_sha256']: raise ValueError('Split manifest differs from pin')
    c=dict(config,pin_sha256=pin_digest)
    initial_identity=evidence.input_identity(split)
    def verify_inputs():
        if digest(pin_path)!=pin_digest: raise ValueError('Pinned execution identity changed')
        _verify_pin(data_dir,pin_path)
        loaded=load_split(data_dir,config['dataset'],config['resample'])
        if evidence.input_identity(loaded)!=initial_identity: raise ValueError('Input identity changed during fitting')
    return _fit(split,out_dir,c,evidence_mode='formal',epochs=1000,batch_size=512,
                resume=resume,verify_inputs=verify_inputs,stop_after_epoch=None)


def fit_synthetic(split,out_dir,config,*,epochs=10,batch_size=7,resume=False,stop_after_epoch=None):
    """Bounded fixture-only API; synthetic completions cannot satisfy formal mode."""
    if type(epochs) is not int or epochs<0 or epochs>30: raise ValueError('Synthetic budget must be integer 0..30')
    if type(batch_size) is not int or batch_size<1: raise ValueError('Positive integer batch size required')
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or not 0<=stop_after_epoch<epochs):
        raise ValueError('Synthetic interruption must precede terminal epoch')
    if type(config.get('n')) is not int or not 1 <= config['n'] <= 6 or type(config.get('T')) is not int or not 0 <= config['T'] <= 8:
        raise ValueError('Synthetic register/query bounds require 1<=n<=6 and 0<=T<=8')
    if sum(len(split[name]['y']) for name in ('fit','validation')) > 128:
        raise ValueError('Synthetic API permits at most 128 rows')
    if any(len(str(split.get(k,'')))==64 for k in ('manifest_sha256','data_sha256','split_sha256')):
        raise ValueError('Official authenticated inputs are forbidden in synthetic mode')
    c=dict(config,synthetic_epochs=epochs,synthetic_batch_size=batch_size)
    return _fit(split,out_dir,c,evidence_mode='synthetic',epochs=epochs,batch_size=batch_size,
                resume=resume,stop_after_epoch=stop_after_epoch)


def _validate_scale(value):
    if (not isinstance(value,numbers.Real) or isinstance(value,(bool,np.bool_))
            or not np.isfinite(value) or value!=1.):
        raise ValueError('Frozen scale must be a finite nonboolean real number equal to one')


def _validate(split,config):
    _validate_scale(config.get('scale'))
    evidence.input_identity(split)  # rejects held-out inputs before directory creation
    if config.get('arm') != 'fixed': raise ValueError('Unknown basis arm')
    if config.get('device','cpu')!='cpu': raise ValueError('Reviewed execution backend is CPU float64')
    if type(config.get('T')) is not int or config['T']<0: raise ValueError('Nonnegative integer T required')
    if type(config.get('n')) is not int or config['n']<1: raise ValueError('Positive integer n required')
    if type(config.get('seed')) is not int or config.get('shuffle_seed')!=config['seed']+1000000:
        raise ValueError('Paired integer initialization and shuffle seeds required')
    mean,scale=np.asarray(split['mean']),np.asarray(split['scale'])
    if mean.ndim!=1 or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Finite fit-only scaler with positive scales required')
    for name in ('fit','validation'):
        s=split[name]; a=np.asarray(s['angles']); y=np.asarray(s['y'])
        if a.ndim!=2 or a.shape[1]!=len(mean) or not len(a) or not np.isfinite(a).all():
            raise ValueError('Nonempty finite angle matrices required')
        if y.shape!=(len(a),) or set(np.unique(y))!={0,1}: raise ValueError('Both binary classes required in each authorized section')
        for key in ('ids','groups'):
            if np.asarray(s[key]).shape!=y.shape: raise ValueError('Input ID/group shape mismatch')
        if len(np.unique(s['ids']))!=len(y): raise ValueError('Duplicate row IDs')
        for key in ('x','standardized'):
            if key in s and (np.asarray(s[key]).shape!=a.shape or not np.isfinite(s[key]).all()):
                raise ValueError('Original feature/preprocessed shape mismatch')
    for key in ('ids','groups'):
        if np.intersect1d(split['fit'][key],split['validation'][key]).size:
            raise ValueError('Fit/validation '+key+' overlap')
    if len(mean)>2**config['n'] or max(1,(len(mean)-1).bit_length())!=config['n']:
        raise ValueError('Register must match preserved initialization dimension')


def _loss(y,values,class_weights):
    return (class_weights[y]*(1-(2*y-1)*values)/2).mean()


def _parameter_artifact(path,theta,probe,features):
    save_npz(path,theta=theta.detach().cpu().numpy(),probe=probe.detach().cpu().numpy())
    write_json(Path(path).with_suffix('.physical.json'),physical_probe(probe.detach().cpu(),features,'diagonal_joint'))


def _spectral_diagnostics(prepared,probe,batch_size):
    accum=[]
    with torch.no_grad():
        count=prepared['phase_powers'].shape[0]
        for offset in range(0,count,batch_size):
            w=model.spectral_weights(model.subset_inputs(prepared,slice(offset,offset+batch_size)),probe)
            entropy=-(w*torch.log(w.clamp_min(torch.finfo(w.dtype).tiny))).sum(-1)
            accum.append(torch.stack((w.square().sum(-1),w.max(-1).values,entropy),dim=-1).cpu().numpy())
    values=np.concatenate(accum)
    return {key:dict(mean=float(values[:,i].mean()),minimum=float(values[:,i].min()),maximum=float(values[:,i].max()))
            for i,key in enumerate(('inverse_participation_ratio','maximum_eigenbasis_weight','spectral_entropy'))}


def _fit(split,out_dir,config,*,evidence_mode,epochs,batch_size,resume,stop_after_epoch,verify_inputs=lambda:None):
    started=time.monotonic(); cpu_started=time.process_time(); out=Path(out_dir)
    _validate(split,config); p=protocol(); torch.set_num_threads(1)
    identity=evidence.identity_for(split,config,evidence_mode,p)
    existing=evidence.begin(out,identity,resume=resume)
    if existing is not None: return existing
    features=split['fit']['angles'].shape[1]
    theta,probe=initial_parameters(features,'diagonal_joint',config['T'],config['seed'])
    theta=torch.nn.Parameter(theta); probe=torch.nn.Parameter(probe)
    optimizer=torch.optim.Adam([dict(params=[theta],lr=.1),dict(params=[probe],lr=.01)])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4,min_lr=[.002,.0002])
    shuffle=np.random.default_rng(config['shuffle_seed'])
    cache_started=time.monotonic()
    prepared={name:model.prepare_inputs(torch.tensor(split[name]['angles'],dtype=torch.float64),config['arm'],
                T=config['T'],scale=1.,n=config['n']) for name in ('fit','validation')}
    cache_seconds=time.monotonic()-cache_started
    cache_bytes=sum(t.numel()*t.element_size() for section in prepared.values() for t in section.values())
    y={name:torch.tensor(split[name]['y'],dtype=torch.long) for name in prepared}
    counts=np.bincount(split['fit']['y'],minlength=2)
    weights=torch.tensor(len(y['fit'])/(2*counts),dtype=torch.float64)
    current=0; history=[]; best_key=None; best_epoch=0; best_parameters=None; gradient_norms={}; diagnostics={}; elapsed_prior=0.; cpu_prior=0.
    if resume and (out/'progress.json').exists():
        progress=json.loads((out/'progress.json').read_text())
        for field in ('checkpoint_file','previous_checkpoint_file'):
            filename=progress.get(field)
            if filename is not None and (not isinstance(filename,str) or Path(filename).name!=filename or not filename.startswith('checkpoint-') or not filename.endswith('.pt')):
                raise ValueError('Invalid checkpoint generation basename')
        checkpoint=out/progress['checkpoint_file']
        if digest(checkpoint)!=progress['checkpoint_sha256']: raise ValueError('Interrupted checkpoint mutated')
        for name,sha in progress['bound_artifacts'].items():
            if digest(out/name)!=sha: raise ValueError('Interrupted artifact mutated: '+name)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity_sha256']!=digest(out/'identity.json'): raise ValueError('Resume identity mismatch')
        if state['epoch']!=progress['epoch'] or state['best_epoch']!=progress['best_epoch']:
            raise ValueError('Published progress/checkpoint generation mismatch')
        selected=min(state['history'],key=lambda h:checkpoint_key(h['validation'][primary_metric(config.get('dataset'))],h['validation_loss'],h['epoch']))
        if selected['epoch']!=state['best_epoch'] or tuple(state['best_key'])!=checkpoint_key(selected['validation'][primary_metric(config.get('dataset'))],selected['validation_loss'],selected['epoch']):
            raise ValueError('Published checkpoint selection/history mismatch')
        with torch.no_grad(): theta.copy_(state['theta']); probe.copy_(state['probe'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        shuffle.bit_generator.state=state['shuffle_rng']; torch.set_rng_state(state['torch_rng'])
        current=state['epoch']; history=state['history']; best_epoch=state['best_epoch']; best_key=tuple(state['best_key'])
        gradient_norms=state['gradient_norms']; diagnostics=state['diagnostics']
        best_parameters=state['best_parameters']
        elapsed_prior=state['elapsed_seconds']; cpu_prior=state['cpu_seconds']
        # Preserve unpublished interrupted attempts once, outside bounded live slots.
        published={progress['checkpoint_file'],progress.get('previous_checkpoint_file')}
        for candidate in list(out.glob('checkpoint-*.pt'))+list(out.glob('.checkpoint-*.pt')):
            if candidate.name not in published:
                archive=out/'interrupted'; archive.mkdir(exist_ok=True)
                target=archive/candidate.name
                if target.exists(): raise ValueError('Interrupted generation archive collision')
                os.replace(candidate,target)
    else:
        if resume and any(path.name!='identity.json' for path in out.iterdir()):
            raise ValueError('No authenticated resumable checkpoint; attempt preserved')
        write_json(out/'protocol.json',p)
        save_npz(out/'preprocessing.npz',mean=split['mean'],scale=split['scale'],
            **{name+'_'+key:split[name][key] for name in ('fit','validation') for key in ('ids','groups','y')})
        save_npz(out/'preprocessed.npz',**{name+'_angles':split[name]['angles'] for name in ('fit','validation')})
        _parameter_artifact(out/'initial.npz',theta,probe,features)
        diagnostics['initial']={name:_spectral_diagnostics(prepared[name],probe,batch_size) for name in prepared}
    def predictions(name):
        blocks=[]
        with torch.no_grad():
            for offset in range(0,len(y[name]),batch_size):
                blocks.append(model.scores_prepared(model.subset_inputs(prepared[name],slice(offset,offset+batch_size)),theta,probe))
        return torch.cat(blocks)
    def gradients():
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(y['fit']),batch_size):
            index=slice(offset,offset+batch_size); size=len(y['fit'][index])
            values=model.scores_prepared(model.subset_inputs(prepared['fit'],index),theta,probe)
            (_loss(y['fit'][index],values,weights)*size/len(y['fit'])).backward()
        return dict(theta=float(torch.linalg.vector_norm(theta.grad)),probe=float(torch.linalg.vector_norm(probe.grad)))
    def verify():
        verify_inputs()
        if evidence.source_hashes()!=identity['source_hashes']: raise ValueError('Scientific source changed during fitting')
        if evidence.input_identity(split)!=identity['inputs']: raise ValueError('In-memory scientific input changed')
    def assess(epoch):
        nonlocal best_key,best_epoch,best_parameters
        verify()
        values={name:predictions(name) for name in prepared}
        losses={name:float(_loss(y[name],values[name],weights)) for name in prepared}
        val=values['validation'].numpy(); threshold=0.
        item=dict(epoch=epoch,train_loss=losses['fit'],validation_loss=losses['validation'],
            train=metrics(split['fit']['y'],values['fit'].numpy(),threshold),
            validation=metrics(split['validation']['y'],val,threshold),learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(item); key=checkpoint_key(item['validation'][primary_metric(config.get('dataset'))],losses['validation'],epoch)
        if best_key is None or key<best_key:
            best_key=key; best_epoch=epoch
            best_parameters=dict(theta=theta.detach().clone(),probe=probe.detach().clone())
            diagnostics['best']={name:_spectral_diagnostics(prepared[name],probe,batch_size) for name in prepared}
        scheduler.step(losses['validation'])  # original epoch-zero assessment is intentional
        if epoch in (0,200,500,1000) or epoch==epochs: gradient_norms[str(epoch)]=gradients()
    def checkpoint():
        state=dict(identity_sha256=digest(out/'identity.json'),epoch=current,theta=theta.detach(),probe=probe.detach(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),shuffle_rng=shuffle.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=None,history=history,best_key=best_key,best_epoch=best_epoch,
            best_parameters=best_parameters,
            gradient_norms=gradient_norms,diagnostics=diagnostics,
            elapsed_seconds=elapsed_prior+time.monotonic()-started,cpu_seconds=cpu_prior+time.process_time()-cpu_started)
        # Each immutable generation embeds its selected parameters and history.
        # Only progress publication makes it authoritative; failed unpublished
        # generations remain as evidence and are never used for resume.
        name=f'checkpoint-{current}-{uuid.uuid4().hex}.pt'; temporary=out/('.'+name)
        torch.save(state,temporary); os.replace(temporary,out/name)
        bound={name:digest(out/name) for name in ('identity.json','protocol.json','preprocessing.npz','preprocessed.npz',
            'initial.npz','initial.physical.json')}
        previous=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        write_json(out/'progress.json',dict(epoch=current,best_epoch=best_epoch,checkpoint_file=name,
            previous_checkpoint_file=previous.get('checkpoint_file'),
            checkpoint_sha256=digest(out/name),bound_artifacts=bound,elapsed_seconds=state['elapsed_seconds']))
        # Publication succeeded: retain current and immediate predecessor only.
        # Delete only the explicitly known obsolete successful generation, never
        # unknown/unpublished snapshots (those are archived on authenticated resume).
        obsolete=previous.get('previous_checkpoint_file')
        if obsolete is not None: (out/obsolete).unlink(missing_ok=True)
        return state
    if not history:
        assess(0); checkpoint()
    if stop_after_epoch is not None and current>=stop_after_epoch:
        return dict(status='interrupted',actual_epochs=current)
    while current<epochs:
        current+=1
        order=shuffle.permutation(len(y['fit']))
        for offset in range(0,len(order),batch_size):
            index=torch.as_tensor(order[offset:offset+batch_size],dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            values=model.scores_prepared(model.subset_inputs(prepared['fit'],index),theta,probe)
            loss=_loss(y['fit'][index],values,weights)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite fitting loss')
            loss.backward(); optimizer.step()
        if current%5==0 or current==epochs: assess(current)
        checkpoint()
        if stop_after_epoch is not None and current>=stop_after_epoch:
            return dict(status='interrupted',actual_epochs=current)
    terminal_gradient=gradients(); _parameter_artifact(out/'terminal.npz',theta,probe,features)
    diagnostics['terminal']={name:_spectral_diagnostics(prepared[name],probe,batch_size) for name in prepared}
    final_state=checkpoint(); torch.save(final_state,out/'checkpoint.pt')
    _parameter_artifact(out/'best.npz',best_parameters['theta'],best_parameters['probe'],features)
    write_json(out/'history.json',history)
    selected=predict_parameters(out/'best.npz',config,split['validation']['angles'])
    threshold=0.
    selected_loss=float(np.mean((len(y['fit'])/(2*counts))[split['validation']['y']]*(1-(2*split['validation']['y']-1)*selected)/2))
    reconstructed=min(history,key=lambda h:checkpoint_key(h['validation'][primary_metric(config.get('dataset'))],h['validation_loss'],h['epoch']))
    if reconstructed['epoch']!=best_epoch or not np.isclose(selected_loss,reconstructed['validation_loss'],atol=1e-12,rtol=0):
        raise ValueError('Independent best-checkpoint reconstruction failed')
    save_npz(out/'validation.npz',ids=split['validation']['ids'],groups=split['validation']['groups'],y=split['validation']['y'],scores=selected)
    write_json(out/'recipe.json',dict(schema='qrenn-a07-multidata-recipe-v1',config=config,source_hashes=identity['source_hashes'],
        parameters='best.npz',preprocessing='preprocessing.npz',threshold=threshold,default_threshold=0.,
        threshold_scope='Fixed zero threshold; no threshold optimization',original_feature_inference=True))
    verify()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    costs=dict(elapsed_seconds=elapsed_prior+time.monotonic()-started,cpu_seconds=cpu_prior+time.process_time()-cpu_started,
        peak_process_rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024),prepared_cache_bytes=cache_bytes,
        prepared_cache_seconds_this_invocation=cache_seconds,torch_threads=1,device='cpu',resume=bool(resume))
    return evidence.seal(out,dict(config=config,best_epoch=best_epoch,actual_epochs=current,cap_hit=current==epochs,
        validation_metrics=metrics(split['validation']['y'],selected,threshold),validation_loss=selected_loss,threshold=threshold,
        gradient_norms=gradient_norms,terminal_gradient_norms=terminal_gradient,diagnostics=diagnostics,costs=costs,
        elapsed_seconds=costs['elapsed_seconds'],device='cpu',torch_threads=1,early_stopping=False))


def predict_parameters(parameters_path,config,angles):
    with np.load(parameters_path,allow_pickle=False) as parameters:
        theta=torch.tensor(parameters['theta'],dtype=torch.float64); probe=torch.tensor(parameters['probe'],dtype=torch.float64)
    blocks=[]
    with torch.no_grad():
        for start in range(0,len(angles),512):
            prepared=model.prepare_inputs(torch.tensor(angles[start:start+512],dtype=torch.float64),config['arm'],
                T=config['T'],scale=config['scale'],n=config['n'])
            blocks.append(model.scores_prepared(prepared,theta,probe).numpy())
    return np.concatenate(blocks) if blocks else np.empty(0,dtype=np.float64)


def predict_original_features(out_dir,original_features,*,evidence_mode='formal'):
    """Authenticate saved evidence and replay using the original feature columns."""
    out=Path(out_dir); evidence.authenticate(out,evidence_mode=evidence_mode)
    recipe=json.loads((out/'recipe.json').read_text())
    with np.load(out/'preprocessing.npz',allow_pickle=False) as preprocessing:
        mean,scale=preprocessing['mean'],preprocessing['scale']
    x=np.asarray(original_features,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(mean) or not np.isfinite(x).all(): raise ValueError('Finite original-feature matrix required')
    return predict_parameters(out/'best.npz',recipe['config'],.5+np.arctan((x-mean)/scale)/np.pi)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-pin',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--pin',type=Path)
    parser.add_argument('--dataset',choices=protocol()['datasets'])
    parser.add_argument('--r',type=int,choices=range(5))
    parser.add_argument('--i',type=int,choices=range(3))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.create_pin:
        create_pin(args.data,args.create_pin); print(json.dumps(dict(status='pinned',path=str(args.create_pin))))
        return
    if any(value is None for value in (args.out,args.pin,args.dataset,args.r,args.i)):
        parser.error('A job requires --out --pin --dataset --r --i')
    config=next(c for c in make_tasks() if (c['dataset'],c['resample'],c['init'])==(args.dataset,args.r,args.i))
    result=fit(args.data,args.out,config,pin_path=args.pin,resume=args.resume)
    print(json.dumps(dict(status='complete',best_epoch=result['best_epoch'],actual_epochs=result['actual_epochs'],
                         validation_metrics=result['validation_metrics'])))


if __name__=='__main__': main()


def phishing_source(data_dir,legacy_data_dir,qrenn_root,resample,init):
    """Authenticate a reused sealed run; return source parameters, never fit it.

    New manifest bytes intentionally differ. All actual original/preprocessed
    arrays, labels, row IDs, groups and scaler must match the old dependency.
    """
    from matched_kernel_study.train import _original_pin, _checkpoint_record
    from dataset_screen.data import load_split as old_load
    from eigenbasis_study.evidence import _array_identity
    original=_original_pin(legacy_data_dir,qrenn_root)
    old=old_load(legacy_data_dir,'phishing',resample)
    new=load_split(data_dir,'phishing',resample)
    for section in ('fit','validation'):
        for key in old[section]:
            if _array_identity(old[section][key])!=_array_identity(new[section][key]):
                raise ValueError('New Phishing input differs from sealed source: '+section+'/'+key)
    for key in ('mean','scale'):
        if not np.array_equal(old[key],new[key]): raise ValueError('Phishing scaler differs')
    record=_checkpoint_record(legacy_data_dir,qrenn_root,resample,init,original,old)
    path=Path(record['path'])
    parameters={}
    for stage in ('initial','best','terminal'):
        with np.load(path/(stage+'.npz'),allow_pickle=False) as z:
            parameters[stage]={k:z[k].copy() for k in ('theta','probe')}
    with np.load(path/'validation.npz',allow_pickle=False) as z:
        predictions={k:z[k].copy() for k in z.files}
    replay=predict_parameters(path/'best.npz',dict(arm='fixed',n=5,T=8,scale=1.),new['validation']['angles'])
    if not np.allclose(replay,predictions['scores'],rtol=0,atol=2e-12):
        raise ValueError('Phishing selected prediction replay mismatch')
    return dict(kind='reused_sealed_phishing_run',new_run=False,source=record,
                new_inputs=evidence.input_identity(new),parameters=parameters,validation=predictions)
