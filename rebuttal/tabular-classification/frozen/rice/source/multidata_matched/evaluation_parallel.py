"""Frozen-model parallel evaluation with predeclared new-campaign tolerances.

Authenticate the explicitly supplied snapshot and sibling pin before held-out
loading. Spawned workers own isolated outputs; only the parent seals completion.
"""
from pathlib import Path
import concurrent.futures
import multiprocessing
import os
import time
import numpy as np
from . import evaluation_numerics as serial
from . import evaluate as original, data, evidence, train, kernels as k, baselines

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
_THREAD_LIMIT = None


def _initialize_worker():
    global _THREAD_LIMIT
    for name in THREAD_ENV:
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(limits=1)
    train.torch.set_num_threads(1)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    data.write_json(temporary, value)
    temporary.replace(path)


def evaluate(data_dir, snapshot, out_dir, workers=8):
    """Evaluate the complete declared formal freeze, with 1..8 worker processes."""
    frozen = original.authenticate_snapshot(snapshot)
    pin_path = Path(data_dir).resolve().parent / 'pin.json'
    train._verify_pin(data_dir, pin_path)
    if data.digest(Path(data_dir) / 'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('Frozen data manifest differs')
    pin_digest = data.digest(pin_path)
    snapshot_digest = data.digest(snapshot)

    def verify_inputs():
        train._verify_pin(data_dir, pin_path)
        if data.digest(pin_path) != pin_digest:
            raise ValueError('Source pin changed during evaluation')
        if data.digest(snapshot) != snapshot_digest:
            raise ValueError('Frozen snapshot changed during evaluation')

    return _evaluate(frozen, out_dir, 'formal', workers,
                     dict(path=str(pin_path), sha256=pin_digest), verify_inputs,
                     data_dir=str(Path(data_dir).resolve()))


def evaluate_synthetic(snapshot, out_dir, splits, workers=2):
    """Bounded fixture entry; cannot authorize formal scientific evidence."""
    frozen = original.authenticate_snapshot(snapshot, mode='synthetic')
    if not 1 <= len(frozen['bundles']) <= 6:
        raise ValueError('Synthetic evaluation bound')
    snapshot_digest = data.digest(snapshot)

    def verify_inputs():
        if data.digest(snapshot) != snapshot_digest:
            raise ValueError('Frozen snapshot changed during evaluation')

    return _evaluate(frozen, out_dir, 'synthetic', workers, None, verify_inputs,
                     splits=splits)


def _evaluate_bundle(record, destination, mode, data_dir=None, split=None):
    """The serial per-bundle calculation, using unchanged inference/audit helpers."""
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    rows, audits, evaluations = [], [], []
    try:
        location = Path(record['path'])
        identity = k.read(location / 'identity.json')
        if split is None:
            split = data.load_split(data_dir, record['coordinate'][1], record['coordinate'][2], include_test=True)
        development = {a: b for a, b in split.items() if a != 'test'}
        if isinstance(development.get('groups'), dict):
            development['groups'] = {a: b for a, b in development['groups'].items() if a != 'test'}
        if evidence.input_identity(development) != identity['inputs']:
            raise ValueError('Evaluation input differs from frozen development data')
        if 'test' not in split:
            raise ValueError('Explicit held-out section required')
        for key in ('ids', 'groups'):
            if any(np.intersect1d(split['test'][key], split[p][key]).size for p in ('fit', 'validation')):
                raise ValueError('Held-out rows/groups overlap development')
        recipe = k.read(location / 'recipe.json')
        for method in record['methods']:
            started = time.monotonic()
            kind = identity['kind']
            name = original._name(record, method)
            predict = baselines.predict_original_features if kind == 'classical' else k.predict_original_features
            accumulation = None
            if kind == 'kernel' or (kind == 'classical' and method == 'rbf_svm'):
                compact = k.arrays(location / 'selected' / f'{method}.npz')
                n = len(compact['dual_coef'])
                unit = np.finfo(float).eps / 2
                dual_l1 = float(np.sum(abs(compact['dual_coef'])))
                accumulation = dict(support_count=n, dual_l1=dual_l1,
                    sequential_dot_roundoff_bound_unit_kernel=float((n * unit / (1 - n * unit)) * dual_l1),
                    bound_role='Diagnostic only; per-dot bound assumes exact kernel entries of magnitude <=1; not the acceptance threshold')
            for partition in ('validation', 'test'):
                section = split[partition]
                scores = predict(location, section['x'], method, evidence_mode=mode)
                native_replay = None
                if partition == 'validation':
                    saved = k.arrays(location / ('validation_' + method + '.npz'))
                    for key in ('ids', 'groups', 'y'):
                        if not np.array_equal(saved[key], section[key]):
                            raise ValueError('Frozen validation rows differ')
                    native_replay = serial.compare_scores(section['y'], saved['scores'], scores, kind=kind, method=method)
                    if not native_replay['accepted']:
                        data.write_json(out / 'failed_replay.json', dict(name=name, partition=partition, stage='production_vs_native', audit=native_replay))
                    serial.require_accepted(native_replay)
                    scores = saved['scores']
                k.save_predictions(out / partition / (name + '.npz'), section, scores)
                independent = original.independent_predict(location, section['x'], method)
                audit = serial.compare_scores(section['y'], scores, independent, kind=kind, method=method)
                audit.update(name=name, partition=partition, bundle_path=str(location), method=method,
                    production_vs_native=native_replay, independent_vs_reference=dict(audit), accumulation=accumulation,
                    verification_kind='classical_portable_replay_same_implementation_not_independent' if kind == 'classical' else 'independent_full_state_numpy_scipy',
                    reference_authority='native_saved_validation' if partition == 'validation' else 'original_feature_production_test',
                    native_test_prediction_available=False)
                audits.append(audit)
                _atomic_json(out / 'replay_audit.json', audits)
                if not audit['accepted']:
                    data.write_json(out / 'failed_replay.json', dict(name=name, partition=partition, stage='independent_vs_reference', audit=audit))
                serial.require_accepted(audit)
                prediction_path = partition + '/' + name + '.npz'
                prediction = k.arrays(out / prediction_path)
                rows.append(dict(partition=partition, dataset=record['coordinate'][1], resample=record['coordinate'][2],
                    init=record['coordinate'][3], method=original._method(record, method), prediction_path=prediction_path,
                    prediction_sha256=data.digest(out / prediction_path), **k.metrics(prediction['y'], prediction['scores'])))
            evaluations.append(dict(name=name, elapsed_seconds=time.monotonic() - started,
                completion_sha256=record['completion_sha256'], support_count=recipe['selected'][method].get('support_count'),
                validation_rows=len(split['validation']['y']), test_rows=len(split['test']['y'])))
        from threadpoolctl import threadpool_info
        execution = dict(pid=os.getpid(), numerical_threads=1, torch_threads=train.torch.get_num_threads(),
                         threadpools=threadpool_info())
        if execution['torch_threads'] != 1 or any(p['num_threads'] != 1 for p in execution['threadpools']):
            raise ValueError('Worker numerical thread limit differs')
        result = dict(rows=rows, audits=audits, evaluations=evaluations, execution=execution)
        _atomic_json(out / 'result.json', result)
        return result
    except Exception as exc:
        _atomic_json(out / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                     completed_selected_models=len(evaluations), prediction_records=len(rows)))
        raise


def _evaluate(frozen, out_dir, mode, workers, pin, verify_inputs, *, data_dir=None, splits=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('Use one through eight worker processes')
    out = Path(out_dir)
    if out.exists():
        raise ValueError('Evaluation attempt immutable; preserve failed attempt and choose a new destination')
    numerical_policy = serial.policy()
    numerical_policy['source_pin'] = pin
    supplement_digest = numerical_policy['supplement_sha256']
    parallel_source = Path(__file__).resolve()
    parallel_digest = data.digest(parallel_source)
    out.mkdir(parents=True, exist_ok=False)
    data.write_json(out / 'freeze.json', frozen)
    data.write_json(out / 'numerical_policy.json', numerical_policy)
    (out / 'evaluation_numerics_source.py').write_bytes(Path(serial.__file__).read_bytes())
    (out / 'evaluation_parallel_source.py').write_bytes(parallel_source.read_bytes())
    execution = dict(schema='qrenn-a07-parallel-execution-v1', worker_count=workers,
                     start_method='spawn', numerical_threads_per_worker=1,
                     runner_source_file='evaluation_parallel_source.py', runner_sha256=parallel_digest,
                     numerical_supplement_sha256=supplement_digest,
                     collection_order='original frozen bundle order, original method order, validation then test',
                     retries=0, bundles=[])
    data.write_json(out / 'parallel_execution.json', execution)
    rows, audits, evaluations, completed = [], [], [], {}
    _atomic_json(out / 'progress.json', dict(completed_selected_models=0, planned=frozen['selected_count']))
    old_env = {name: os.environ.get(name) for name in THREAD_ENV}
    try:
        # Spawned interpreters inherit one-thread settings before importing NumPy.
        for name in THREAD_ENV:
            os.environ[name] = '1'
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker) as pool:
            futures = {}
            for index, record in enumerate(frozen['bundles']):
                section = None if splits is None else splits[(record['coordinate'][1], record['coordinate'][2])]
                future = pool.submit(_evaluate_bundle, record, str(out / 'workers' / f'{index:05d}'), mode, data_dir, section)
                futures[future] = index
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                _atomic_json(out / 'progress.json', dict(completed_selected_models=sum(len(v['evaluations']) for v in completed.values()),
                             completed_bundles=len(completed), planned=frozen['selected_count']))
        # Canonical outputs are assembled only after every worker succeeded.
        for index, record in enumerate(frozen['bundles']):
            result = completed[index]
            owned = out / 'workers' / f'{index:05d}'
            for row in result['rows']:
                target = out / row['prediction_path']
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError('Duplicate prediction destination')
                (owned / row['prediction_path']).replace(target)
            rows.extend(result['rows'])
            audits.extend(result['audits'])
            evaluations.extend(result['evaluations'])
            execution['bundles'].append(dict(index=index, coordinate=record['coordinate'], **result['execution']))
        if len(evaluations) != frozen['selected_count'] or len(rows) != 2 * frozen['selected_count'] or len(audits) != len(rows):
            raise ValueError('Incomplete parallel evaluation')
        if sum(a['production_vs_native'] is not None for a in audits) != frozen['selected_count']:
            raise ValueError('Incomplete native validation replay')
        data.write_json(out / 'parallel_execution.json', execution)
        data.write_json(out / 'replay_audit.json', audits)
        result = original.aggregate(rows)
        result.update(selected_model_count=len(evaluations), model_costs=[dict(coordinate=r['coordinate'],
            costs=k.read(Path(r['path']) / 'completion.json').get('costs', {}),
            resources=k.read(Path(r['path']) / 'completion.json').get('resources', {})) for r in frozen['bundles']],
            evaluation_costs=evaluations, numerical_policy_file='numerical_policy.json',
            original_score_atol=serial.ORIGINAL_ATOL, svm_score_atol=serial.SVM_ATOL)
        original._csv(out / 'runs.csv', rows)
        original._csv(out / 'resample_means.csv', result['resample_means'])
        original._csv(out / 'table.csv', result['summary'])
        original._csv(out / 'paired_differences.csv', result['paired_differences'])
        data.write_json(out / 'summary.json', result)
        verify_inputs()
        current_policy = serial.policy()
        current_policy['source_pin'] = pin
        if current_policy != numerical_policy or data.digest(parallel_source) != parallel_digest:
            raise ValueError('Evaluation supplement changed during execution')
        for record in frozen['bundles']:
            evidence.authenticate(record['path'], evidence_mode=mode)
            if data.digest(Path(record['path']) / 'completion.json') != record['completion_sha256']:
                raise ValueError('Model changed during evaluation')
        completion = dict(schema='qrenn-a07-multidata-evaluation-v1', evidence_mode=mode,
            snapshot_sha256=frozen['snapshot_sha256'], selected_model_count=len(evaluations),
            numerical_policy_sha256=data.digest(out / 'numerical_policy.json'), supplement_sha256=supplement_digest,
            parallel_runner_sha256=parallel_digest, parallel_execution_sha256=data.digest(out / 'parallel_execution.json'),
            artifacts=evidence.inventory(out, exclude=('completion.json',)))
        data.write_json(out / 'independent_audit.json', original._audit_evaluation(out, completion))
        completion['artifacts'] = evidence.inventory(out, exclude=('completion.json',))
        data.write_json(out / 'completion.json', completion)
        return result
    except Exception as exc:
        _atomic_json(out / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
            completed_selected_models=sum(len(v['evaluations']) for v in completed.values()),
            prediction_records=sum(len(v['rows']) for v in completed.values()),
            original_score_atol=serial.ORIGINAL_ATOL, svm_score_atol=serial.SVM_ATOL))
        raise
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
