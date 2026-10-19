"""Exact source/input/artifact authentication for frozen matched kernels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dataset_screen.data import ROOT, digest, write_json
from dataset_screen.evaluate import runtime
from eigenbasis_study.evidence import (source_hashes as original_sources,
    input_identity as original_input_identity, _array_identity as array_identity)


def source_hashes():
    result=original_sources()
    for path in sorted((ROOT/'matched_kernel_study').rglob('*')):
        if path.name=='protocol.json' or (path.suffix=='.py' and not path.name.startswith('test_')):
            if not path.resolve().is_relative_to(ROOT):raise ValueError('Scientific source outside checkout')
            result[str(path.relative_to(ROOT))]=digest(path)
    for name in result:
        if name.endswith('.py'):
            module_name=name[:-3].replace('/','.').removesuffix('.__init__')
            module=sys.modules.get(module_name)
            if module is not None and Path(module.__file__).resolve()!=(ROOT/name).resolve():
                raise ValueError('Loaded scientific source outside frozen closure: '+name)
    return dict(sorted(result.items()))


def _contains_test(value):
    if isinstance(value,dict):return any(str(k).lower().startswith('test') or _contains_test(v) for k,v in value.items())
    if isinstance(value,(list,tuple)):return any(_contains_test(v) for v in value)
    return False


def input_identity(split):
    if _contains_test(split):raise ValueError('Fitting inputs must exclude test sections and arrays')
    return original_input_identity(split)


def inventory(out):
    out=Path(out)
    result={}
    for path in sorted(out.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink artifacts are not allowed')
        if path.is_file() and path!=out/'completion.json':result[str(path.relative_to(out))]=digest(path)
    return result


def begin(out,identity):
    out=Path(out)
    if out.exists():
        if not (out/'completion.json').exists():raise ValueError('Unsealed existing attempt preserved; choose a new directory')
        result=authenticate(out,evidence_mode=identity['evidence_mode'],expected_config=identity['config'])
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Existing evidence identity differs')
        return result
    out.mkdir(parents=True)
    write_json(out/'identity.json',identity)
    return None


def seal(out,record):
    out=Path(out)
    if (out/'completion.json').exists():raise ValueError('Completion is immutable')
    identity=json.loads((out/'identity.json').read_text())
    if identity['source_hashes']!=source_hashes():raise ValueError('Scientific source changed during fitting')
    result=dict(record,schema='qrenn-a07-matched-kernel-completion-v1',evidence_mode=identity['evidence_mode'],
        identity_sha256=digest(out/'identity.json'),artifacts=inventory(out))
    write_json(out/'completion.json',result)
    return result


def authenticate(out_dir,expected_config=None,evidence_mode='formal'):
    out=Path(out_dir)
    try:
        result=json.loads((out/'completion.json').read_text());identity=json.loads((out/'identity.json').read_text())
    except (OSError,ValueError) as exc:raise ValueError('Missing or invalid completion') from exc
    if result.get('schema')!='qrenn-a07-matched-kernel-completion-v1':raise ValueError('Completion schema mismatch')
    if result.get('evidence_mode')!=evidence_mode or identity.get('evidence_mode')!=evidence_mode:raise ValueError('Evidence mode mismatch')
    if digest(out/'identity.json')!=result['identity_sha256']:raise ValueError('Identity mutated')
    if identity['source_hashes']!=source_hashes():raise ValueError('Scientific source identity changed')
    if expected_config is not None and identity['config']!=expected_config:raise ValueError('Configuration mismatch')
    if result['config']!=identity['config']:raise ValueError('Completion configuration mismatch')
    actual=inventory(out)
    if set(actual)!=set(result['artifacts']):raise ValueError('Artifact inventory changed')
    if actual!=result['artifacts']:raise ValueError('Scientific artifact mutated')
    recipe=json.loads((out/'recipe.json').read_text())
    if recipe['source_hashes']!=identity['source_hashes']:raise ValueError('Recipe source mismatch')
    if json.loads((out/'protocol.json').read_text())!=identity['protocol']:raise ValueError('Protocol artifact mismatch')
    # External path names are provenance. Formal fit/reuse verifies those live
    # inputs before reaching begin(); transferred completed evidence is portable.
    return result
