"""Source, input and artifact authentication for the paired development study."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from dataset_screen.data import ROOT, digest, source_hashes as original_sources, write_json
from dataset_screen.evaluate import json_hash, runtime, save_npz


def source_hashes():
    from .data import source_hashes as collect
    result=collect()
    for name in result:
        if name.endswith('.py'):
            module=sys.modules.get(name[:-3].replace('/','.').removesuffix('.__init__'))
            if module is not None and Path(module.__file__).resolve()!=(ROOT/name).resolve():
                raise ValueError('Loaded source outside frozen closure')
    return result


def _contains_test(value):
    if isinstance(value,dict):
        return any(str(k).lower().startswith('test') or _contains_test(v) for k,v in value.items())
    if isinstance(value,(list,tuple)): return any(_contains_test(v) for v in value)
    return False


def input_identity(split):
    """Bind actual authorized arrays, including order, dtype, labels and multiplicity."""
    if _contains_test(split): raise ValueError('Fitting inputs must exclude test sections and arrays')
    result={}
    for name in ('mean','scale'):
        result[name]=_array_identity(split[name])
    for name in ('fit','validation'):
        result[name]={k:_array_identity(v) for k,v in sorted(split[name].items())}
    for name in ('manifest_sha256','data_sha256','split_sha256'):
        result[name]=split[name]
    return result


def _array_identity(value):
    value=np.asarray(value)
    if value.dtype.hasobject: raise ValueError('Object arrays cannot identify scientific inputs')
    return dict(dtype=value.dtype.str,shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def identity_for(split,config,mode,protocol):
    return dict(schema='qrenn-a07-multidata-identity-v1',evidence_mode=mode,
                config=config,protocol=protocol,source_hashes=source_hashes(),
                inputs=input_identity(split),runtime=runtime())


def begin(out,identity,*,resume=False):
    out=Path(out)
    # Validate identity before creating an attempt directory.
    if out.exists() and (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=identity:
            raise ValueError('Existing evidence identity differs')
        if (out/'completion.json').exists():
            return authenticate(out,evidence_mode=identity['evidence_mode'],expected_config=identity['config'])
        if not resume: raise ValueError('Incomplete attempt preserved; exact resume or a new directory required')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Unidentified existing attempt preserved')
    else:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'identity.json',identity)
    return None


def inventory(out,*,exclude=()):
    out=Path(out)
    if any(p.is_symlink() for p in out.rglob('*')): raise ValueError('Symlink artifact refused')
    return {str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*'))
            if path.is_file() and str(path.relative_to(out)) not in exclude}


def seal(out,record):
    out=Path(out)
    if (out/'completion.json').exists(): raise ValueError('Completion is immutable')
    identity=json.loads((out/'identity.json').read_text())
    if identity['source_hashes']!=source_hashes(): raise ValueError('Scientific source changed during fitting')
    result=dict(record,schema='qrenn-a07-multidata-completion-v1',evidence_mode=identity['evidence_mode'],
                identity_sha256=digest(out/'identity.json'),artifacts=inventory(out,exclude=('completion.json',)))
    write_json(out/'completion.json',result)
    return result


def authenticate(out,*,evidence_mode='formal',expected_config=None):
    out=Path(out)
    try:
        completion=json.loads((out/'completion.json').read_text())
        identity=json.loads((out/'identity.json').read_text())
    except (ValueError,OSError) as exc: raise ValueError('Missing or invalid completion') from exc
    if completion.get('evidence_mode')!=evidence_mode or identity.get('evidence_mode')!=evidence_mode:
        raise ValueError('Evidence mode mismatch')
    if digest(out/'identity.json')!=completion['identity_sha256']: raise ValueError('Identity mutated')
    if identity['source_hashes']!=source_hashes(): raise ValueError('Scientific source identity changed')
    if expected_config is not None and identity['config']!=expected_config: raise ValueError('Configuration mismatch')
    current=inventory(out,exclude=('completion.json',))
    if set(current)!=set(completion['artifacts']): raise ValueError('Artifact inventory changed')
    if current!=completion['artifacts']: raise ValueError('Scientific artifact mutated')
    recipe=json.loads((out/'recipe.json').read_text())
    if recipe['source_hashes']!=identity['source_hashes']: raise ValueError('Recipe source mismatch')
    return completion
