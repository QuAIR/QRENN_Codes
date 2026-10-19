"""Authenticated inference and held-out evaluation. This module imports no fitting code."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
import scipy
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, roc_auc_score)

from .data import digest, load_split, protocol, source_hashes, write_json
from .model import scores
from .quantum import fidelity_kernel, iqp_features


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__,joblib=joblib.__version__, platform=platform.platform())


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def balanced_threshold(y, values):
    """Literal unique-score rule: maximum BA, minimum |t|, then higher t."""
    y = np.asarray(y); values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(y) or not len(y) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty matching validation vectors required')
    if set(np.unique(y)) != {0, 1}: raise ValueError('Both classes required')
    # Equivalent to evaluating every candidate, with O(N log N) sorting.
    order = np.argsort(values, kind='stable'); v, labels = values[order], y[order]
    unique, first = np.unique(v, return_index=True)
    positives = np.r_[0, np.cumsum(labels)]
    negatives = np.r_[0, np.cumsum(1-labels)]
    p, n = positives[-1], negatives[-1]
    ba = .5*((p-positives[first])/p+negatives[first]/n)
    candidates = np.r_[unique, np.nextafter(unique[-1], np.inf)]
    quality = np.r_[ba, .5]
    return float(min(zip(candidates, quality), key=lambda pair: (-pair[1], abs(pair[0]), -pair[0]))[0])


def metrics(y, values, threshold):
    y, values = np.asarray(y), np.asarray(values)
    if values.shape != y.shape or not np.isfinite(values).all(): raise ValueError('Invalid scores')
    return dict(roc_auc=float(roc_auc_score(y, values)), average_precision=float(average_precision_score(y, values)),
                accuracy=float(accuracy_score(y, values >= 0)),
                balanced_accuracy=float(balanced_accuracy_score(y, values >= 0)),
                validation_balanced_threshold_accuracy=float(accuracy_score(y, values >= threshold)),
                validation_balanced_threshold_balanced_accuracy=float(balanced_accuracy_score(y, values >= threshold)),
                threshold=float(threshold))


def identity_for(split, config, evidence_mode):
    return dict(schema='qrenn-a07-identity-v1', evidence_mode=evidence_mode, config=config,
                manifest_sha256=split['manifest_sha256'], data_sha256=split['data_sha256'],
                split_sha256=split['split_sha256'], raw_inputs=split.get('raw_inputs',{'kind':'synthetic_fixture'}),
                source_hashes=source_hashes(), runtime=runtime())


def begin(out_dir, identity, *, resume=False):
    """Refuse different evidence; preserve interrupted attempts unless exact resume requested."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    identity_path = out/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity: raise ValueError('Existing evidence identity differs')
        if (out/'completion.json').exists():
            return authenticate(out, evidence_mode=identity['evidence_mode'])
        if not resume: raise ValueError('Incomplete attempt preserved; choose a new directory or exact resume')
    elif any(out.iterdir()):
        raise ValueError('Unidentified existing attempt preserved')
    else:
        write_json(identity_path, identity)
    return None


def seal(out_dir, record):
    """Atomic completion binding every artifact, including recipe and optimizer checkpoint."""
    out = Path(out_dir)
    if (out/'completion.json').exists(): raise ValueError('Completion is immutable')
    identity = json.loads((out/'identity.json').read_text())
    if identity['source_hashes'] != source_hashes(): raise ValueError('Source changed during execution')
    artifacts = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'completion.json'}
    result = dict(record, schema='qrenn-a07-completion-v1', evidence_mode=identity['evidence_mode'],
                  identity_sha256=digest(out/'identity.json'), artifacts=artifacts)
    write_json(out/'completion.json', result)
    return result


def authenticate(out_dir, *, evidence_mode='formal', expected_config=None):
    out = Path(out_dir)
    try:
        result = json.loads((out/'completion.json').read_text())
        identity = json.loads((out/'identity.json').read_text())
    except (OSError, ValueError) as exc: raise ValueError('Missing or invalid completion') from exc
    if result.get('evidence_mode') != evidence_mode or identity.get('evidence_mode') != evidence_mode:
        raise ValueError('Evidence mode mismatch')
    if digest(out/'identity.json') != result['identity_sha256']: raise ValueError('Identity mutated')
    if identity['source_hashes'] != source_hashes(): raise ValueError('Source identity changed')
    if expected_config is not None and identity['config'] != expected_config: raise ValueError('Configuration mismatch')
    actual = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'completion.json'}
    if actual != set(result['artifacts']): raise ValueError('Artifact inventory changed')
    for name, sha in result['artifacts'].items():
        if digest(out/name) != sha: raise ValueError('Artifact mutated: '+name)
    recipe = json.loads((out/'recipe.json').read_text())
    if recipe['source_hashes'] != identity['source_hashes']: raise ValueError('Recipe source mismatch')
    return result


def gaussian(left, right, gamma):
    distances = np.maximum(np.square(left).sum(1)[:,None] + np.square(right).sum(1)[None,:] - 2*left@right.T, 0.)
    return np.exp(-gamma*distances)


def native_scores(estimator, x):
    if hasattr(estimator, 'decision_function'): return np.asarray(estimator.decision_function(x), dtype=float)
    return 2*np.asarray(estimator.predict_proba(x)[:,list(estimator.classes_).index(1)], dtype=float)-1


def predict_recipe(out_dir, original_features, *, evidence_mode='formal'):
    """Replay original-feature inference with authenticated fit-only preprocessing."""
    out = Path(out_dir); authenticate(out, evidence_mode=evidence_mode)
    recipe = json.loads((out/'recipe.json').read_text())
    x = np.asarray(original_features, dtype=np.float64)
    with np.load(out/'preprocessing.npz', allow_pickle=False) as pre:
        mean, scale = pre['mean'], pre['scale']
    if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x).all(): raise ValueError('Invalid original features')
    z = (x-mean)/scale; angles = .5+np.arctan(z)/np.pi
    if recipe['kind'] == 'qrenn':
        with np.load(out/'best.npz', allow_pickle=False) as params:
            theta, probe = torch.tensor(params['theta']), torch.tensor(params['probe'])
        cfg = recipe['config']; blocks=[]
        with torch.no_grad():
            for start in range(0,len(x),512):
                blocks.append(scores(torch.tensor(angles[start:start+512]), theta, probe, cfg['family'], cfg['scale']).numpy())
        return np.concatenate(blocks) if blocks else np.empty(0)
    saved_runtime=json.loads((out/'identity.json').read_text())['runtime']
    if any(saved_runtime[k]!=runtime()[k] for k in ('numpy','scipy','sklearn','joblib')):
        raise ValueError('Native estimator requires its recorded runtime versions; use a separately audited portable export')
    estimator = joblib.load(out/'selected.joblib')  # authenticated bytes before deserialization
    if recipe['kind'] == 'quantum':
        cfg = recipe['selected_candidate']
        with np.load(out/'kernel_fit.npz', allow_pickle=False) as fitting: fit_angles = fitting['angles']
        if recipe['method'] == 'fidelity':
            design = fidelity_kernel(angles, fit_angles, cfg['scale'], cfg['power'])
        else:
            design = gaussian(iqp_features(angles,cfg['scale'],cfg['power']),
                              iqp_features(fit_angles,cfg['scale'],cfg['power']),cfg['gamma'])
    else: design = angles if recipe['representation'] == 'angles' else z
    return native_scores(estimator, design)


def gap_summary(y, qrenn, comparator, groups, *, replicates=2000, seed=92017):
    """Average AUC over starts, then resamples; common group-bootstrap weights."""
    y=np.asarray(y); q=np.asarray(qrenn); c=np.asarray(comparator); groups=np.asarray(groups)
    if q.shape[:2] != (5,3) or c.shape[0] != 5 or q.shape[-1] != len(y) or c.shape[-1] != len(y):
        raise ValueError('Expected [5,3,N] QRENN and [5,N] comparator scores')
    paired = [float(np.mean([roc_auc_score(y,s) for s in q[r]])-roc_auc_score(y,c[r])) for r in range(5)]
    unique, inverse = np.unique(groups, return_inverse=True); rng=np.random.default_rng(seed); draws=[]
    for _ in range(replicates):
        weights=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inverse]
        if any(weights[y==label].sum()==0 for label in (0,1)): continue
        draws.append(float(np.mean([np.mean([roc_auc_score(y,s,sample_weight=weights) for s in q[r]])-
                                   roc_auc_score(y,c[r],sample_weight=weights) for r in range(5)])))
    if not draws: raise ValueError('No two-class bootstrap replicates')
    return dict(mean_gap=float(np.mean(paired)), paired_resample_gaps=paired,
                sample_sd=float(np.std(paired,ddof=1)), descriptive_gap_at_least_minus_001=bool(np.mean(paired)>=-.01),
                bootstrap_95_percentile=np.percentile(draws,[2.5,97.5]).tolist(), bootstrap_valid_replicates=len(draws),
                bootstrap_replicates=replicates, bootstrap_seed=seed, formal_noninferiority_test=False,
                scope='Conditional on one held-out partition and frozen selected configurations; excludes selection uncertainty')


def authenticate_freeze(out_dir, data_dir, *, full=True):
    """Authenticate complete frozen inventory BEFORE any held-out loading."""
    out=Path(out_dir)
    try:
        frozen=json.loads((out/'freeze.json').read_text()); receipt=json.loads((out/'freeze-completion.json').read_text())
    except (OSError,ValueError) as exc: raise ValueError('Successful freeze required') from exc
    if digest(out/'freeze.json') != receipt['freeze_sha256'] or frozen['evidence_mode']!='formal':
        raise ValueError('Freeze mutated or synthetic')
    pin=json.loads((out/'pin.json').read_text())
    if digest(out/'pin.json') != frozen['pin_sha256'] or pin['sources']!=source_hashes(): raise ValueError('Frozen pin/source mismatch')
    if digest(Path(data_dir)/'manifest.json') != pin['manifest_sha256']: raise ValueError('Frozen manifest changed')
    for name, sha in frozen['bound_files'].items():
        if digest(out/name)!=sha: raise ValueError('Frozen selection/completion changed: '+name)
    if len(frozen['models'])!=90 or frozen['expected_prediction_files']!=90: raise ValueError('Incomplete frozen model inventory')
    if full:
        for name in frozen['models']: authenticate(out/name)
    return frozen


def score_frozen(data_dir, out_dir):
    """The sole formal test entrypoint. All fitting imports are absent."""
    out=Path(out_dir); frozen=authenticate_freeze(out,data_dir)
    target=out/'test'
    if target.exists(): raise ValueError('Existing test attempt preserved')
    target.mkdir(); results={}; vectors={}
    for name, item in frozen['models'].items():
        # Reauthenticate the freeze throughout evaluation, then explicitly open test.
        authenticate_freeze(out,data_dir,full=False)
        split=load_split(data_dir,item['dataset'],item['resample'],include_test=True)
        if split['manifest_sha256']!=frozen['manifest_sha256']: raise ValueError('Manifest changed during test')
        section=split['test']; pred=predict_recipe(out/name,section['x'],evidence_mode='formal')
        filename=name.replace('/','__')+'.npz'
        save_npz(target/filename, ids=section['ids'],y=section['y'],groups=section['groups'],scores=pred)
        results[name]=metrics(section['y'],pred,item['threshold']); vectors[name]=pred
    summaries={}; gaps={}
    for dataset in protocol()['datasets']:
        q=np.asarray([[vectors[f'final/{dataset}/r{r}/i{i}'] for i in range(3)] for r in range(5)])
        selected=[frozen['strong_quantum'][dataset][str(r)] for r in range(5)]
        c=np.asarray([vectors[name] for name in selected])
        split=load_split(data_dir,dataset,0,include_test=True)
        settings=protocol()['evaluation']
        gaps[dataset]=gap_summary(split['test']['y'],q,c,split['test']['groups'],
                                  replicates=settings['bootstrap_replicates'],seed=settings['bootstrap_seed'])
        methods={'qrenn':[[f'final/{dataset}/r{r}/i{i}' for i in range(3)] for r in range(5)]}
        for method in ('lr_standardized','lr_angles','rbf_svm','hist_gradient_boosting','fidelity','projected'):
            methods[method]=[[f'baselines/{dataset}/r{r}/{method}'] for r in range(5)]
        methods['selected_strong_quantum']=[[name] for name in selected]
        summaries[dataset]={}
        for method, rows in methods.items():
            summary={}
            for metric in results[rows[0][0]]:
                if metric=='threshold': continue
                values=[float(np.mean([results[name][metric] for name in row])) for row in rows]
                summary[metric]=dict(resamples=values, mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))
            summaries[dataset][method]=summary
    authenticate_freeze(out,data_dir)
    report=dict(schema='qrenn-a07-test-v1',evidence_mode='formal',freeze_sha256=digest(out/'freeze.json'),
                predictions=results, summaries=summaries,gaps=gaps,prediction_files=90,
                statistical_unit='Five development resamples conditional on one common test partition per dataset',
                quantum_computation='Exact classical simulation; training/search costs are not equalized')
    write_json(target/'report.json',report)
    artifacts={str(p.relative_to(target)):digest(p) for p in sorted(target.iterdir()) if p.is_file()}
    write_json(target/'completion.json',dict(freeze_sha256=digest(out/'freeze.json'),artifacts=artifacts))
    return report
