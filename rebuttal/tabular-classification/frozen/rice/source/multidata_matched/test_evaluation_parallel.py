"""Bounded spawned-process differential and failure tests; synthetic inputs only."""
from pathlib import Path
import copy
import numpy as np
import pytest
from . import evaluation_parallel as parallel
from . import evaluation_numerics as serial
from multidata_matched import evaluate as original, evidence, kernels as k, baselines, train
from multidata_matched.test_comparison import split, development


def test_parallel_matches_serial_and_rejects_worker_failure(split, tmp_path):
    dev = development(split)
    cfg = dict(dataset='htru2', resample=0, init=0)
    checkpoint = tmp_path / 'checkpoint'
    train.fit_synthetic(dev, checkpoint, dict(arm='fixed', T=2, n=2, scale=1., seed=82000,
        shuffle_seed=1082000, device='cpu', **cfg), epochs=1, batch_size=8)
    q = tmp_path / 'qrenn'
    k.import_qrenn_synthetic(dev, q, checkpoint, cfg)
    kernel = tmp_path / 'kernel'
    params = k.arrays(checkpoint / 'initial.npz')
    k.fit_synthetic(dev, kernel, params['theta'], params['probe'], dict(cfg, arm='initial'))
    classic = tmp_path / 'classical'
    baselines.fit_synthetic(dev, classic, dict(dataset='htru2', resample=0))
    bundles = [q, kernel, classic]
    snapshot = tmp_path / 'freeze.json'
    original.freeze_synthetic(bundles, snapshot)
    before = {str(p): evidence.inventory(p) for p in bundles}
    source_before = evidence.source_hashes()
    serial_sha = parallel.data.digest(Path(serial.__file__))
    splits = {('htru2', 0): split}
    serial_out, parallel_out = tmp_path / 'serial', tmp_path / 'parallel'
    expected = original.evaluate_synthetic(snapshot, serial_out, splits)
    actual = parallel.evaluate_synthetic(snapshot, parallel_out, splits, workers=2)
    # Timing is the sole differing scientific summary field.
    for result in (expected, actual):
        for item in result['evaluation_costs']:
            item.pop('elapsed_seconds')
    for key in ('selected_model_count','summary','resample_means','paired_differences','model_costs','evaluation_costs'):
        assert actual[key] == expected[key]
    for name in ('runs.csv', 'resample_means.csv', 'table.csv', 'paired_differences.csv',
                 'freeze.json'):
        assert (parallel_out / name).read_bytes() == (serial_out / name).read_bytes(), name
    predictions = list(parallel_out.glob('*/*.npz'))
    assert len(predictions) == 14
    for path in predictions:
        relative = path.relative_to(parallel_out)
        assert path.read_bytes() == (serial_out / relative).read_bytes()
    assert original.audit_evaluation(parallel_out)['predictions'] == 14
    audits = k.read(parallel_out / 'replay_audit.json')
    assert len(audits) + sum(a['production_vs_native'] is not None for a in audits) == 21
    assert all(a['accepted'] and a['classification_disagreements'] == 0 for a in audits)
    execution = k.read(parallel_out / 'parallel_execution.json')
    assert execution['worker_count'] == 2 and execution['start_method'] == 'spawn'
    assert execution['retries'] == 0
    assert execution['numerical_supplement_sha256'] == serial_sha
    assert [r['index'] for r in execution['bundles']] == [0, 1, 2]
    for item in execution['bundles']:
        assert item['torch_threads'] == 1
        assert all(pool['num_threads'] == 1 for pool in item['threadpools'])
    completion = k.read(parallel_out / 'completion.json')
    assert completion['parallel_runner_sha256'] == parallel.data.digest(Path(parallel.__file__))
    assert completion['parallel_execution_sha256'] == parallel.data.digest(parallel_out / 'parallel_execution.json')
    assert (parallel_out / 'evaluation_numerics_source.py').read_bytes() == Path(serial.__file__).read_bytes()
    assert before == {str(p): evidence.inventory(p) for p in bundles}
    assert source_before == evidence.source_hashes()
    assert serial_sha == parallel.data.digest(Path(serial.__file__))
    with pytest.raises(ValueError, match='immutable'):
        parallel.evaluate_synthetic(snapshot, parallel_out, splits)
    for count in (0, 9, True):
        with pytest.raises(ValueError, match='worker'):
            parallel.evaluate_synthetic(snapshot, tmp_path / f'bad-workers-{count}', splits, workers=count)
    # Pass immutable input-identity authentication, then fail inside a real worker.
    overlap = copy.deepcopy(split)
    overlap['test']['groups'][0] = overlap['fit']['groups'][0]
    failed = tmp_path / 'worker-failed'
    with pytest.raises(ValueError, match='overlap'):
        parallel.evaluate_synthetic(snapshot, failed, {('htru2', 0): overlap}, workers=2)
    assert (failed / 'failure.json').exists() and not (failed / 'completion.json').exists()
    assert list((failed / 'workers').glob('*/failure.json'))
    # Parent final identity verification is also a sealing gate after workers finish.
    changed = tmp_path / 'changed-during-evaluation'
    def reject():
        raise ValueError('Source pin changed during evaluation')
    frozen = original.authenticate_snapshot(snapshot, mode='synthetic')
    with pytest.raises(ValueError, match='Source pin changed'):
        parallel._evaluate(frozen, changed, 'synthetic', 2, None, reject, splits=splits)
    assert (changed / 'failure.json').exists() and not (changed / 'completion.json').exists()
    bad = k.read(snapshot)
    bad['source_hashes']['multidata_matched/evaluate.py'] = 'bad'
    bad['snapshot_sha256'] = original.json_hash({a: b for a, b in bad.items() if a != 'snapshot_sha256'})
    bad_snapshot = tmp_path / 'bad-freeze.json'
    parallel.data.write_json(bad_snapshot, bad)
    with pytest.raises(ValueError, match='source/protocol'):
        parallel.evaluate_synthetic(bad_snapshot, tmp_path / 'bad-source', splits)
    assert not (tmp_path / 'bad-source').exists()
    selected = kernel / 'selected' / 'projected.npz'
    saved_bytes = selected.read_bytes()
    selected.unlink()
    with pytest.raises(ValueError, match='inventory'):
        parallel.evaluate_synthetic(snapshot, tmp_path / 'missing-model', splits)
    assert not (tmp_path / 'missing-model').exists()
    selected.write_bytes(saved_bytes)
    (kernel / 'selected' / 'projected.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='mutated'):
        parallel.evaluate_synthetic(snapshot, tmp_path / 'bad-model', splits)
    assert not (tmp_path / 'bad-model').exists()


def test_formal_gates_precede_pool_and_test_loading(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Held-out loading/pool began before authentication')
    monkeypatch.setattr(parallel.data, 'load_split', forbidden)
    monkeypatch.setattr(parallel.concurrent.futures, 'ProcessPoolExecutor', forbidden)
    def reject_snapshot(*args, **kwargs):
        raise ValueError('Invalid frozen snapshot')
    monkeypatch.setattr(parallel.original, 'authenticate_snapshot', reject_snapshot)
    with pytest.raises(ValueError, match='Invalid frozen snapshot'):
        parallel.evaluate(tmp_path, tmp_path / 'snapshot', tmp_path / 'invalid')
    monkeypatch.setattr(parallel.original, 'authenticate_snapshot', lambda p: dict(snapshot_sha256='newly-frozen'))
    def reject_pin(*args):
        raise ValueError('Pinned scientific source mismatch')
    monkeypatch.setattr(parallel.train, '_verify_pin', reject_pin)
    with pytest.raises(ValueError, match='Pinned'):
        parallel.evaluate(tmp_path, tmp_path / 'snapshot', tmp_path / 'pin-failed')
    assert not (tmp_path / 'invalid').exists()
    assert not (tmp_path / 'pin-failed').exists()
