"""Run each original source snapshot in a separate Python process."""
from pathlib import Path
import argparse, copy, csv, json, sys, warnings
from collections import Counter

HERE = Path(__file__).resolve().parent
METRICS = ('roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy')

def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))

def main():
    if not __debug__: raise RuntimeError('Run verification without Python optimization flags')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'replay', 'train'])
    p.add_argument('--campaign', choices=['phishing-htru2', 'rice'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--dataset', choices=['phishing', 'htru2', 'rice'])
    p.add_argument('--all', action='store_true')
    p.add_argument('--resample', type=int, default=0); p.add_argument('--init', type=int, default=0)
    p.add_argument('--out', type=Path); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, str(HERE/'frozen'/a.campaign/'source'))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from multidata_matched import data, evidence, evaluate, kernels as k, train
    torch.set_num_threads(1); threadpool_limits(1)
    root = a.data/a.campaign
    subset = json.loads((root/'subset.json').read_text())
    assert subset['source_hashes'] == evidence.source_hashes(), 'Source snapshot differs'
    assert data.digest(root/'data/manifest.json') == subset['manifest_sha256']
    if a.dataset and a.dataset not in subset['datasets']: raise ValueError('Dataset outside this subset')
    if a.command == 'train':
        trainer, input_dir, pin_path = train, root/'data', root/'pin.json'
        if a.dataset == 'phishing':
            from eigenbasis_study import train as trainer
            input_dir, pin_path = root/'legacy-phishing/data', root/'legacy-phishing/pin.json'
        config = next(x for x in trainer.make_tasks() if x['dataset'] == a.dataset and x['resample'] == a.resample and x['init'] == a.init and x['arm'] == 'fixed')
        trainer._verify_pin(input_dir, pin_path)
        trainer.load_split(input_dir, a.dataset, a.resample)
        if a.out.exists(): raise ValueError('Use a fresh training output directory')
        if a.dry_run:
            print(json.dumps({'training_not_executed': True, 'config': config, 'inputs_and_pin_verified': True})); return
        trainer.fit(input_dir, a.out, config, pin_path=pin_path); return

    expected = set()
    for d in subset['datasets']:
        for r in range(5):
            expected.add(('classical', d, r, None, None))
            for i in range(3):
                expected.add(('qrenn', d, r, i, None))
                for arm in ['initial', 'trained']: expected.add(('kernel', d, r, i, arm))
    assert {tuple(x['coordinate']) for x in subset['bundles']} == expected
    assert len(subset['bundles']) == len(expected)
    assert sum(len(x['methods']) for x in subset['bundles']) == subset['selected_count']
    rows = read_csv(root/'evaluation/runs.csv')
    table = read_csv(root/'evaluation/table.csv')
    for filename, native_rows in [('runs.csv', rows), ('table.csv', table), ('resample_means.csv', read_csv(root/'evaluation/resample_means.csv'))]:
        published = [{key: value for key, value in row.items() if key != 'campaign'}
                     for row in read_csv(HERE/'records'/filename) if row['campaign'] == a.campaign]
        assert published == native_rows, 'Published CSV differs from archive: '+filename
    splits, expected_predictions = {}, set()
    def get_split(d, r):
        if (d, r) not in splits:
            split = data.load_split(root/'data', d, r, include_test=True)
            for left, right in [('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
                for key in ['ids', 'groups']: assert not set(split[left][key]) & set(split[right][key])
            splits[d, r] = split
        return splits[d, r]
    verified_bundles, checks, replayed, angle_hash_differences = 0, 0, [], 0
    replay_warnings = Counter()
    for record in subset['bundles']:
        kind, d, r, i, arm = record['coordinate']
        out = root/record['path']
        identity = k.read(out/'identity.json'); recipe = k.read(out/'recipe.json')
        assert data.digest(out/'completion.json') == record['completion_sha256']
        assert data.digest(out/'identity.json') == record['identity_sha256']
        assert list(evaluate._coordinates(identity)) == record['coordinate']
        assert sorted(recipe['selected']) == record['methods']
        split = get_split(d, r)
        development = {key: value for key, value in split.items() if key != 'test'}
        development['groups'] = {key: value for key, value in split['groups'].items() if key != 'test'}
        actual_inputs = evidence.input_identity(development)
        recorded_inputs = copy.deepcopy(identity['inputs'])
        # libm/NumPy arctangent bytes can differ across platforms. Authenticate
        # all stored raw/scaler/standardized arrays exactly; retain and report
        # the derived angle hash differences, then check prediction tolerances.
        for part in ['fit', 'validation']:
            actual_hash = actual_inputs[part]['angles'].pop('sha256')
            recorded_hash = recorded_inputs[part]['angles'].pop('sha256')
            angle_hash_differences += actual_hash != recorded_hash
        assert actual_inputs == recorded_inputs
        evaluate.check_candidates(out)
        verified_bundles += 1
        for method in record['methods']:
            for partition in ['validation', 'test']:
                name = partition+'/'+evaluate._name(record, method)+'.npz'
                expected_predictions.add(name)
                if a.command != 'replay' or a.dataset and d != a.dataset: continue
                if not a.all and (r != 0 or i not in [None, 0]): continue
                saved = k.arrays(root/'evaluation'/name)['scores']
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter('always', RuntimeWarning)
                    score = evaluate.independent_predict(out, split[partition]['x'], method)
                replay_warnings.update(str(w.message) for w in observed)
                discrepancy = evaluate.discrepancy(saved, score)
                tolerance = 2e-8 if kind == 'kernel' or method == 'rbf_svm' else 2e-10
                assert discrepancy['max_abs_score_error'] <= tolerance, (name, discrepancy)
                assert discrepancy['classification_disagreements'] == 0, (name, discrepancy)
                replayed.append(dict(prediction=name, tolerance=tolerance, **discrepancy))
    assert {x['prediction_path'] for x in rows} == expected_predictions
    assert len(rows) == len(expected_predictions) == 2*subset['selected_count']
    for row in rows:
        path = root/'evaluation'/row['prediction_path']
        assert data.digest(path) == row['prediction_sha256']
        pred = k.arrays(path); split = get_split(row['dataset'], int(row['resample']))[row['partition']]
        for key in ['ids', 'groups', 'y']: assert np.array_equal(pred[key], split[key])
        for metric, value in k.metrics(pred['y'], pred['scores']).items():
            assert abs(value - float(row[metric])) < 1e-12
            row[metric] = value; checks += 1
    computed = evaluate.aggregate(rows)
    for item in computed['summary']:
        target = next(t for t in table if all(t[key] == item[key] for key in ['partition', 'dataset', 'method']))
        for metric in METRICS:
            for suffix in ['mean', 'sample_sd']:
                key = metric+'_'+suffix
                assert abs(item[key]-float(target[key])) < 1e-12
    print(json.dumps({'campaign': a.campaign, 'bundles_verified': verified_bundles,
                      'prediction_records': len(rows), 'metric_checks': checks,
                      'derived_angle_hash_differences': angle_hash_differences,
                      'replayed_predictions': len(replayed),
                      'largest_replay_score_error': max((x['max_abs_score_error'] for x in replayed), default=None),
                      'replay_classification_disagreements': sum(x['classification_disagreements'] for x in replayed),
                      'replay_rank_disagreements': sum(x['ranking_disagreements'] for x in replayed),
                      'runtime_warning_counts': dict(replay_warnings)}, indent=2))

if __name__ == '__main__': main()
