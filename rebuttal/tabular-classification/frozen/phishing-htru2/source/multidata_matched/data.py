"""Verified official inputs, duplicate-preserving splits and fit-only transforms.

Preparation is deliberately offline: callers stage the official archives first.
Loading authenticates parsed data and split files against the immutable manifest.
The default interface exposes no held-out test row IDs, labels or predictions.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'magic': dict(archive='159.zip', member='magic04.data', sha256='252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d', rows=19020, features=10, positive=12332),
    'htru2': dict(archive='372.zip', member='HTRU_2.csv', sha256='ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379', rows=17898, features=8, positive=1639),
    'phishing': dict(archive='327.zip', member='Training Dataset.arff',
        url='https://archive.ics.uci.edu/static/public/327/phishing%2Bwebsites.zip',
        sha256='514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940',
        rows=11055, features=30, positive=4898, groups=5785, conflicting_groups=64, conflicting_rows=357),
    'spambase': dict(archive='94.zip', member='spambase.data',
        url='https://archive.ics.uci.edu/static/public/94/spambase.zip',
        sha256='813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636',
        rows=4601, features=57, positive=1813, groups=4207, conflicting_groups=3, conflicting_rows=6),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    """Atomically replace a strict JSON artifact (never allow NaN/Infinity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def protocol():
    """Return a fresh mapping of the literal, declared protocol settings."""
    return json.loads((Path(__file__).parent/'protocol.json').read_text())


def source_hashes():
    """Hash complete repository-local production source closure, sorted by path.

    Every non-test dataset_screen module is an entrypoint. AST local-import
    traversal includes unchanged lower-level modules, executing parent package
    initializers and their own local dependencies. Third-party runtime versions
    belong in execution identity.
    """
    pending = list((ROOT/'multidata_matched').rglob('*.py'))
    pending = [p for p in pending if not p.name.startswith('test_')]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        # Importing a.b.module executes each existing parent __init__.py.
        # Traverse their imports too, including dependencies reached only here.
        parent = path.parent
        while parent != ROOT and parent.is_relative_to(ROOT):
            initializer = parent/'__init__.py'
            if initializer.is_file() and initializer not in seen:
                pending.append(initializer)
            parent = parent.parent
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
                base = ROOT
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                else:
                    base = ROOT
                prefix = (node.module+'.') if node.module else ''
                modules = ([node.module] if node.module else []) + [prefix+n.name for n in node.names]
            else:
                continue
            for module in modules:
                target = base.joinpath(*module.split('.'))
                for candidate in (target.with_suffix('.py'), target/'__init__.py'):
                    if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
                        pending.append(candidate)
    for package in {p.parent for p in seen}:
        if (package/'protocol.json').is_file(): seen.add(package/'protocol.json')
    seen.add(ROOT/'multidata_matched/protocol.json')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(seen)}


def _save_npz(path, arrays):
    """Immutable array artifact: reuse only byte-independent equal contents."""
    path = Path(path)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as old:
                matches = set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items())
        except (ValueError, OSError):
            matches = False
        if not matches:
            raise ValueError('Existing array artifact differs: '+str(path))
        return
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _archive(raw_dir, name, spec):
    wanted = raw_dir/spec['archive']
    if wanted.is_file():
        source = wanted
    else:
        # Accept user-staged official filename or a checksum-identical alias.
        aliases = [p for p in raw_dir.glob('*.zip') if digest(p) == spec['sha256']]
        if len(aliases) != 1:
            raise ValueError('Expected exactly one staged official archive: '+name)
        source = aliases[0]
    if digest(source) != spec['sha256']:
        raise ValueError('Official archive checksum mismatch: '+name)
    with zipfile.ZipFile(source) as archive:
        members = [p for p in archive.namelist() if Path(p).name == spec['member']]
        if len(members) != 1:
            raise ValueError('Expected unique archive member: '+spec['member'])
        content = archive.read(members[0])
    return content, hashlib.sha256(content).hexdigest(), source


def _parse(content, name, spec):
    if spec['member'].lower().endswith('.arff'):
        raw, _ = arff.loadarff(io.StringIO('\n'.join(content.decode('utf-8-sig').splitlines())))
        # Nominal ARFF columns arrive as bytes; converting each preserves order.
        columns = [np.asarray(raw[field], dtype=np.float64) for field in raw.dtype.names]
        array = np.column_stack(columns)
    else:
        if name == 'magic':
            rows = list(csv.reader(io.StringIO('\n'.join(content.decode('utf-8-sig').splitlines()))))
            array = np.array([[float(v) for v in row[:-1]]+[1. if row[-1]=='g' else 0. if row[-1]=='h' else float('nan')] for row in rows])
        else:
            array = np.loadtxt(io.StringIO('\n'.join(content.decode('utf-8-sig').splitlines())), delimiter=',', dtype=np.float64)
    if array.shape != (spec['rows'], spec['features']+1) or not np.isfinite(array).all():
        raise ValueError('Unexpected row/feature shape or nonfinite official data: '+name)
    labels = array[:, -1]
    if name == 'phishing':
        if not np.isin(labels, [-1, 1]).all():
            raise ValueError('Unexpected Phishing labels')
        y = (labels == -1).astype(np.int64)
    else:
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Expected binary original labels')
        y = labels.astype(np.int64)
    if int(y.sum()) != spec['positive']:
        raise ValueError('Unexpected positive label count: '+name)
    return np.ascontiguousarray(array[:, :-1]), y


def _groups(x, y):
    _, groups = np.unique(x, axis=0, return_inverse=True)
    counts = np.bincount(groups)
    positives = np.bincount(groups, weights=y).astype(np.int64)
    conflicts = (positives > 0) & (positives < counts)
    return groups, counts, positives, (2*positives > counts).astype(np.int64), conflicts


def prepare(raw_dir, out_dir):
    """Offline prepare all four official archives with one shared outer test/dataset.

    Row order, feature order, all multiplicities and conflicting labels survive.
    A final manifest is atomic; an existing differing manifest is never replaced.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = protocol()['split']
    manifest = dict(schema='qrenn-a07-multidata-data-v1', datasets={}, splits=[],
        protocol_sha256=digest(Path(__file__).parent/'protocol.json'), split_protocol=settings,
        angle_mapping='0.5+arctan((x-mean_fit)/scale_fit)/pi',
        group_policy='exact_feature_vectors; majority_stratification_tie_zero; preserve_all_rows_and_labels')
    for name, spec in DATASETS.items():
        content, member_hash, archive_path = _archive(raw_dir, name, spec)
        x, y = _parse(content, name, spec)
        groups, counts, positives, majority, conflicts = _groups(x, y)
        observed = dict(groups=len(counts), conflicting_groups=int(conflicts.sum()), conflicting_rows=int(counts[conflicts].sum()))
        if any(observed[k] != spec[k] for k in observed if k in spec):
            raise ValueError('Unexpected duplicate/conflict counts: '+name+' '+str(observed))
        target = out_dir/(name+'.npz')
        original_labels = np.where(y==1, -1, 1) if name=='phishing' else np.where(y==1,'g','h') if name=='magic' else y.copy()
        _save_npz(target, dict(x=x, y=y, original_labels=original_labels, groups=groups, group_counts=counts,
            group_positive_counts=positives, group_majority_labels=majority, conflicting_groups=np.flatnonzero(conflicts)))
        manifest['datasets'][name] = dict(spec, **{k:v for k,v in observed.items() if k not in spec},
            unique_feature_groups=len(counts), member_sha256=member_hash, parsed_sha256=digest(target),
            raw_archive_path=os.path.relpath(archive_path.resolve(), out_dir.resolve()))
        ids = np.arange(len(counts))
        development, test = train_test_split(ids, test_size=settings['test_fraction'],
            random_state=settings['outer_seed'], stratify=majority)
        for split, seed in enumerate(settings['validation_seeds']):
            fit, validation = train_test_split(development, test_size=settings['validation_fraction_of_development'],
                random_state=seed, stratify=majority[development])
            row_ids = {k+'_ids': np.flatnonzero(np.isin(groups, value)) for k, value in
                [('fit', fit), ('validation', validation), ('test', test)]}
            scaler = StandardScaler().fit(x[row_ids['fit_ids']])
            path = out_dir/f'{name}-split-{split}.npz'
            _save_npz(path, dict(row_ids, mean=scaler.mean_, scale=scaler.scale_))
            manifest['splits'].append(dict(dataset=name, split=split, seed=seed,
                outer_seed=settings['outer_seed'], archive_sha256=digest(path),
                counts={k[:-4]: dict(total=len(v), positive=int(y[v].sum()), groups=len(np.unique(groups[v]))) for k,v in row_ids.items()}))
    target = out_dir/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Existing preparation manifest differs; preserve attempt and use a new destination')
    if not target.exists():
        write_json(target, manifest)
    return manifest


def load_split(data_dir, dataset, split, include_test=False):
    """Authenticated fit/validation sections; test rows only by explicit opt-in.

    Each section: x,standardized,angles float64[N,d]; y,ids,groups int64[N].
    Metadata: mean/scale float64[d], groups maps authorized sections to IDs,
    manifest_sha256, data_sha256, split_sha256, dataset, split.
    """
    if not isinstance(dataset, str) or not dataset or Path(dataset).name != dataset or type(split) is not int or split < 0:
        raise ValueError('Dataset basename and nonnegative integer split required')
    if type(include_test) is not bool:
        raise ValueError('include_test must be boolean')
    folder = Path(data_dir)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'qrenn-a07-multidata-data-v1' or manifest.get('protocol_sha256') != digest(Path(__file__).parent/'protocol.json'):
        raise ValueError('Preparation schema/protocol identity mismatch')
    if dataset not in manifest['datasets']:
        raise ValueError('Unknown prepared dataset')
    raw_archive = folder/manifest['datasets'][dataset]['raw_archive_path']
    if not raw_archive.is_file() or digest(raw_archive) != manifest['datasets'][dataset]['sha256']:
        raise ValueError('Official raw archive checksum mismatch or missing archive')
    entries = [e for e in manifest['splits'] if e['dataset'] == dataset and e['split'] == split]
    if len(entries) != 1:
        raise ValueError('Expected unique prepared split')
    raw_path, part_path = folder/(dataset+'.npz'), folder/f'{dataset}-split-{split}.npz'
    if digest(raw_path) != manifest['datasets'][dataset]['parsed_sha256'] or digest(part_path) != entries[0]['archive_sha256']:
        raise ValueError('Prepared dataset/split checksum mismatch')
    with np.load(raw_path, allow_pickle=False) as raw, np.load(part_path, allow_pickle=False) as part:
        names = ['fit', 'validation']+(['test'] if include_test else [])
        result = dict(mean=part['mean'].copy(), scale=part['scale'].copy(), groups={},
            manifest_sha256=digest(manifest_path), data_sha256=digest(raw_path), split_sha256=digest(part_path), dataset=dataset, split=split)
        for section in names:
            ids = part[section+'_ids']
            x = raw['x'][ids].copy()
            z = (x-result['mean'])/result['scale']
            result[section] = dict(x=x, standardized=z, angles=.5+np.arctan(z)/np.pi,
                y=raw['y'][ids].copy(), ids=ids.copy(), groups=raw['groups'][ids].copy())
            result['groups'][section] = result[section]['groups'].copy()
    return result


def verify_legacy_splits(data_dir, authority):
    """Require exact old row IDs and scaler bytes for both shared datasets."""
    for dataset in ('phishing', 'spambase'):
        for r in range(5):
            with np.load(Path(data_dir)/f'{dataset}-split-{r}.npz') as new, np.load(Path(authority)/f'{dataset}-split-{r}.npz') as old:
                if set(new.files)!=set(old.files) or any(not np.array_equal(new[k],old[k]) for k in old.files):
                    raise ValueError('Legacy split/preprocessing mismatch')
    return True


if __name__ == '__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--raw',required=True);cli.add_argument('--out',required=True);cli.add_argument('--legacy',required=True)
    args=cli.parse_args();prepare(args.raw,args.out);verify_legacy_splits(args.out,args.legacy)
