"""Independent complete classical and exactly simulated quantum comparator grids."""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import load_split, protocol, source_hashes, write_json
from .evaluate import (balanced_threshold, begin, gaussian, identity_for, json_hash,
                       metrics, native_scores, save_npz, seal)
from .quantum import fidelity_kernel, iqp_features

CLASSICAL=('lr_standardized','lr_angles','rbf_svm','hist_gradient_boosting')
QUANTUM=('fidelity','projected')


def candidate_grids():
    p=protocol(); c=p['classical']; q=p['quantum']
    grids={method:[dict(C=value) for value in c['LR']['C']] for method in CLASSICAL[:2]}
    grids['rbf_svm']=[dict(C=C,gamma_multiplier=g) for C,g in itertools.product(c['RBF_SVM']['C'],c['RBF_SVM']['gamma_multipliers'])]
    grids['hist_gradient_boosting']=[dict(learning_rate=lr,max_leaf_nodes=n,l2_regularization=l2)
        for lr,n,l2 in itertools.product(c['HistGradientBoosting']['learning_rate'],c['HistGradientBoosting']['max_leaf_nodes'],c['HistGradientBoosting']['l2_regularization'])]
    grids['fidelity']=[dict(power=r,scale=s,C=C) for r,s,C in itertools.product(q['powers'],q['scales'],q['C'])]
    grids['projected']=[dict(power=r,scale=s,C=C,gamma_multiplier=g)
        for r,s,C,g in itertools.product(q['powers'],q['scales'],q['C'],q['projected_gamma_multipliers'])]
    return grids


def fit(data_dir,out_dir,config):
    c=dict(config)
    if c.get('dataset') not in protocol()['datasets'] or c.get('resample') not in range(5) or c.get('method') not in CLASSICAL+QUANTUM:
        raise ValueError('Invalid formal baseline coordinate')
    split=load_split(data_dir,c['dataset'],c['resample'])
    if split['manifest_sha256']!=c.get('manifest_sha256'): raise ValueError('Pinned manifest mismatch')
    split['raw_inputs']=json.loads((Path(data_dir)/'manifest.json').read_text())['datasets'][c['dataset']]
    if c.get('source_hashes_sha256',json_hash(source_hashes()))!=json_hash(source_hashes()): raise ValueError('Pinned sources differ')
    def verify_inputs():
        loaded=load_split(data_dir,c['dataset'],c['resample'])
        if any(loaded[k]!=split[k] for k in ('manifest_sha256','data_sha256','split_sha256')): raise ValueError('Input identity changed')
    return _fit(split,out_dir,c,candidate_grids()[c['method']],'formal',verify_inputs)


def fit_synthetic(split,out_dir,method,candidates):
    if method not in CLASSICAL+QUANTUM or not 1<=len(candidates)<=3: raise ValueError('Bounded explicit synthetic grid required')
    c=dict(dataset=split['dataset'],resample=split['split'],method=method,synthetic_candidates=candidates)
    return _fit(split,out_dir,c,candidates,'synthetic',lambda:None)


@threadpool_limits.wrap(limits=1)
def _fit(split,out_dir,config,grid,mode,verify_inputs):
    started=time.monotonic(); torch.set_num_threads(1); out=Path(out_dir)
    if 'test' in split: raise ValueError('Fitting input must exclude held-out test')
    identity=identity_for(split,config,mode); existing=begin(out,identity)
    if existing is not None: return existing
    (out/'candidates').mkdir()
    save_npz(out/'preprocessing.npz',mean=split['mean'],scale=split['scale'],fit_ids=split['fit']['ids'],validation_ids=split['validation']['ids'],
             fit_groups=split['fit']['groups'],validation_groups=split['validation']['groups'],fit_y=split['fit']['y'],validation_y=split['validation']['y'])
    method=config['method']; quantum=method in QUANTUM; features=split['fit']['x'].shape[1]
    y=split['fit']['y']; validation_y=split['validation']['y']; counts=np.bincount(y,minlength=2)
    if not (counts>0).all(): raise ValueError('Both fitting classes required')
    sample_weights=(len(y)/(2*counts))[y]
    representation='angles' if method=='lr_angles' else 'standardized'
    best=None; selected_estimator=None; records=[]; cache_key=None; design=None; validation_design=None
    for index,raw in enumerate(grid):
        verify_inputs()
        if identity['source_hashes']!=source_hashes(): raise ValueError('Sources changed during baseline search')
        candidate=dict(raw); tick=time.monotonic()
        if method in CLASSICAL[:2]:
            estimator=LogisticRegression(C=candidate['C'],solver='lbfgs',max_iter=5000,tol=1e-8,class_weight='balanced')
        elif method=='rbf_svm':
            candidate['gamma']=candidate['gamma_multiplier']/features
            estimator=SVC(C=candidate['C'],kernel='rbf',gamma=candidate['gamma'],tol=.001,class_weight='balanced',cache_size=512,max_iter=-1)
        elif method=='hist_gradient_boosting':
            estimator=HistGradientBoostingClassifier(**candidate,max_iter=300,early_stopping=False,random_state=39011+config['resample'])
        else:
            key=(candidate['power'],candidate['scale'],candidate.get('gamma_multiplier'))
            if key!=cache_key:
                a,b=split['fit']['angles'],split['validation']['angles']
                if method=='fidelity':
                    device=config.get('kernel_device','cpu')
                    design=fidelity_kernel(a,a,candidate['scale'],candidate['power'],device=device)
                    validation_design=fidelity_kernel(b,a,candidate['scale'],candidate['power'],device=device)
                else:
                    fit_features=iqp_features(a,candidate['scale'],candidate['power']); val_features=iqp_features(b,candidate['scale'],candidate['power'])
                    denominator=3*features*float(np.var(fit_features,ddof=0))
                    if denominator<=0: raise ValueError('Zero fitting projected-feature variance')
                    gamma=candidate['gamma_multiplier']/denominator
                    design=gaussian(fit_features,fit_features,gamma); validation_design=gaussian(val_features,fit_features,gamma)
                cache_key=key
            if method=='projected': candidate['gamma']=candidate['gamma_multiplier']/(3*features*float(np.var(iqp_features(split['fit']['angles'],candidate['scale'],candidate['power']),ddof=0)))
            estimator=SVC(C=candidate['C'],kernel='precomputed',tol=.001,cache_size=512,max_iter=-1,class_weight='balanced')
        if not quantum:
            design=split['fit'][representation]; validation_design=split['validation'][representation]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(design,y,**({'sample_weight':sample_weights} if method=='hist_gradient_boosting' else {}))
        prediction=native_scores(estimator,validation_design)
        auc=float(roc_auc_score(validation_y,prediction)); key=(-auc,index)
        record=dict(index=index,candidate=candidate,validation_roc_auc=auc,elapsed_seconds=time.monotonic()-tick,
                    warnings=[str(w.message) for w in caught],
                    actual_iterations=np.asarray(getattr(estimator,'n_iter_',[])).tolist())
        records.append(record)
        save_npz(out/'candidates'/f'{index:03d}.npz',ids=split['validation']['ids'],y=validation_y,groups=split['validation']['groups'],scores=prediction)
        write_json(out/'candidates'/f'{index:03d}.json',record)
        if best is None or key<best:
            best=key; selected_estimator=estimator; selected_candidate=candidate; selected_prediction=prediction.copy()
    # Reconstruct from every retained prediction, without relying on incremental state.
    rebuilt=min(records,key=lambda record:(-float(roc_auc_score(validation_y,np.load(out/'candidates'/f"{record['index']:03d}.npz")['scores'])),record['index']))
    if rebuilt['index']!=best[1]: raise ValueError('Baseline selection reconstruction failed')
    threshold=balanced_threshold(validation_y,selected_prediction)
    joblib.dump(selected_estimator,out/'selected.joblib',compress=3)
    if quantum:
        save_npz(out/'kernel_fit.npz',angles=split['fit']['angles'],ids=split['fit']['ids'],y=y)
    if hasattr(selected_estimator,'support_'):
        save_npz(out/'selected_svm.npz',support=selected_estimator.support_,dual_coef=selected_estimator.dual_coef_,
                 intercept=selected_estimator.intercept_,support_vectors=selected_estimator.support_vectors_,classes=selected_estimator.classes_)
    save_npz(out/'validation.npz',ids=split['validation']['ids'],y=validation_y,groups=split['validation']['groups'],scores=selected_prediction)
    write_json(out/'candidate_grid.json',dict(method=method,candidates=grid,records=records,selection='descending validation ROC-AUC, first grid index'))
    write_json(out/'recipe.json',dict(schema='qrenn-a07-recipe-v1',kind='quantum' if quantum else 'classical',method=method,
                                     config=config,source_hashes=identity['source_hashes'],selected_candidate=selected_candidate,
                                     representation=representation,threshold=threshold,original_feature_inference=True,
                                     estimator='selected.joblib',svm_arrays='selected_svm.npz' if hasattr(selected_estimator,'support_') else None,
                                     kernel_fit='kernel_fit.npz' if quantum else None,preprocessing='preprocessing.npz',
                                     runtime_requirement='Recorded numpy/scipy/sklearn/joblib versions for native estimator replay',
                                     computation='exact_classical_simulation' if quantum else 'classical_native_objective'))
    verify_inputs()
    return seal(out,dict(kind='quantum' if quantum else 'classical',method=method,config=config,candidate_count=len(grid),
                         selected_index=best[1],selected_candidate=selected_candidate,threshold=threshold,
                         validation_metrics=metrics(validation_y,selected_prediction,threshold),elapsed_seconds=time.monotonic()-started,
                         kernel_device=config.get('kernel_device','cpu'),torch_threads=torch.get_num_threads(),
                         equal_training_or_search_cost_claim=False))
