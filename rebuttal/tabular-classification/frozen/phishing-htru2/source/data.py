"""Official A07 inputs, immutable grouped partitions and fitting-only transforms."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = {
    'magic': dict(archive='159.zip', member='magic04.data', sha256='252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d', rows=19020, features=10, positive=12332),
    'htru2': dict(archive='372.zip', member='HTRU_2.csv', sha256='ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379', rows=17898, features=8, positive=1639),
}
SPLIT_SEEDS = [17001, 17002, 17003, 17004, 17005]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(raw_dir, destination):
    raw_dir, destination = Path(raw_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'datasets': {}, 'splits': [], 'angle_mapping': '0.5 + arctan(fit-standardized feature)/pi radians'}
    for name, spec in DATASETS.items():
        source = raw_dir / spec['archive']
        if digest(source) != spec['sha256']:
            raise ValueError('Official input checksum mismatch: ' + name)
        with zipfile.ZipFile(source) as z:
            content = z.read(spec['member']).decode()
        raw = np.loadtxt(io.StringIO('\n'.join(content.splitlines())), delimiter=',', dtype=str)
        x = raw[:, :-1].astype(np.float64)
        y = (raw[:, -1] == 'g').astype(np.int64) if name == 'magic' else raw[:, -1].astype(np.int64)
        assert x.shape == (spec['rows'], spec['features'])
        assert y.sum() == spec['positive'] and np.isfinite(x).all()
        _, first, group = np.unique(x, axis=0, return_index=True, return_inverse=True)
        assert np.all(y == y[first[group]]), 'Conflicting duplicate labels need a different policy'
        group_ids = np.arange(len(first))
        group_y = y[first]
        target = destination / (name + '.npz')
        if target.exists():
            with np.load(target) as old:
                assert np.array_equal(old['x'], x) and np.array_equal(old['y'], y)
        else:
            np.savez_compressed(target, x=x, y=y, group=group)
        manifest['datasets'][name] = dict(spec, unique_feature_groups=len(first), parsed_sha256=digest(target))
        for split, seed in enumerate(SPLIT_SEEDS):
            pool, test_g = train_test_split(group_ids, test_size=.2, random_state=seed, stratify=group_y)
            fit_g, val_g = train_test_split(pool, test_size=.25, random_state=seed + 100, stratify=group_y[pool])
            ids = {k: np.flatnonzero(np.isin(group, g)) for k, g in [('fit', fit_g), ('validation', val_g), ('test', test_g)]}
            assert len(np.unique(np.concatenate(list(ids.values())))) == len(y)
            assert sum(map(len, ids.values())) == len(y)
            for a, b in [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
                assert not np.intersect1d(group[ids[a]], group[ids[b]]).size
            transform = StandardScaler().fit(x[ids['fit']])
            filename = destination / f'{name}-split-{split}.npz'
            arrays = dict(ids, mean=transform.mean_, scale=transform.scale_)
            if filename.exists():
                with np.load(filename) as old:
                    assert all(np.array_equal(old[k], v) for k, v in arrays.items())
            else:
                np.savez_compressed(filename, **arrays)
            manifest['splits'].append(dict(dataset=name, split=split, seed=seed, archive_sha256=digest(filename),
                counts={k: {'total': len(idx), 'positive': int(y[idx].sum())} for k, idx in ids.items()}))
    write_json(destination / 'manifest.json', manifest)
    return manifest


def load_split(folder, dataset, split, include_test=False):
    """Model selection deliberately receives no test rows or test labels."""
    folder = Path(folder)
    with np.load(folder / (dataset + '.npz')) as raw, np.load(folder / f'{dataset}-split-{split}.npz') as part:
        result = {}
        for key in ['fit', 'validation'] + (['test'] if include_test else []):
            idx = part[key]
            original = raw['x'][idx]
            z = (original - part['mean']) / part['scale']
            result[key] = dict(x=original.copy(), standardized=z, angles=.5 + np.arctan(z)/np.pi, y=raw['y'][idx].copy(), ids=idx.copy())
        result['mean'] = part['mean'].copy()
        result['scale'] = part['scale'].copy()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.raw, args.out), indent=2))
