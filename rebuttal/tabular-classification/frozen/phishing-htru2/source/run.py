"""Fit/validation-only QRENN calibration and final fits for the A07 protocol."""
from pathlib import Path
import argparse
import itertools
import json
import time

import numpy as np
import torch

from data import digest, load_split, write_json
from metrics import primary, validation_thresholds
import model
from provenance import fit_identity, begin_attempt, verify_complete, completion_fields, attempt_id, record_failure

SOURCE_FILES = ('model.py', 'run.py', 'data.py', 'metrics.py', 'provenance.py')
ARTIFACTS = ('config.json', 'provenance.json', 'attempt.json', 'parameters.npz',
             'validation.npz', 'history.json', 'best.npz', 'current.pt', 'progress.json')


def identity_for(data_dir, config, data):
    counts = np.bincount(data['fit']['y'], minlength=2)
    cw = len(data['fit']['y']) / (2 * counts) if config['weighted'] else np.ones(2)
    identity = fit_identity(data_dir, config['dataset'], config['split'], config, SOURCE_FILES, cw)
    identity['runtime']['torch'] = torch.__version__
    return identity, cw


def initial_parameters(n, slots, seed, probe_family):
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    theta = 2 * torch.pi * torch.rand((slots+1, 3), generator=g, dtype=torch.float64)
    probe = torch.zeros(n, dtype=torch.float64) if probe_family == 'zero' else torch.full((n,), torch.pi/2, dtype=torch.float64)
    if probe_family == 'ry':
        probe = 2 * torch.pi * torch.rand((n,), generator=g, dtype=torch.float64)
    return theta, probe


def probability_loss(score, y, weights):
    return (weights * (1 - (2*y - 1)*score) / 2).mean()


def fit(data_dir, out_dir, config):
    previous = attempt_id(out_dir)
    try:
        return _fit(data_dir, out_dir, config)
    except BaseException as error:
        record_failure(out_dir, previous, error)
        raise


def verify_qrenn_complete(data_dir, out_dir, config):
    """Validate a completed fit without reading held-out test rows."""
    loaded = load_split(data_dir, config['dataset'], config['split'])
    identity, _ = identity_for(data_dir, config, loaded)
    return verify_complete(out_dir, identity, ARTIFACTS)


def _fit(data_dir, out_dir, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_split(data_dir, config['dataset'], config['split'])
    identity, cw = identity_for(data_dir, config, data)
    if (out/'complete.json').exists():
        return verify_complete(out, identity, ARTIFACTS)
    begin_attempt(out, identity, data_dir)
    torch.set_num_threads(int(config.get('threads', 1)))
    device = torch.device(config.get('device', 'cpu'))
    x = {k: torch.as_tensor(data[k]['angles'], dtype=torch.float64, device=device) for k in ['fit','validation']}
    y = {k: torch.as_tensor(data[k]['y'], dtype=torch.float64, device=device) for k in x}
    weights = {k: torch.as_tensor(cw[data[k]['y']], dtype=torch.float64, device=device) for k in x}
    theta0, probe0 = initial_parameters(x['fit'].shape[1], config['T'], config['seed'], config['probe'])
    if 'fixed_probe' in config:
        probe0 = torch.tensor(config['fixed_probe'], dtype=torch.float64)
    theta = torch.nn.Parameter(theta0.to(device).clone())
    learn_probe = config['stage'] == 'calibration' and config['probe'] == 'ry'
    probe = torch.nn.Parameter(probe0.to(device).clone(), requires_grad=learn_probe)
    params = [theta, probe] if learn_probe else [theta]
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=4, min_lr=.002)
    cached = {} if learn_probe else {k: model.moments(x[k], probe.detach(), config['encoding'], config['T']).detach() for k in x}
    shuffle = torch.Generator(device='cpu').manual_seed(config['seed'] + 1000000)
    preparation_seconds = time.perf_counter() - started
    history, best, best_key, stale, updates = [], None, None, 0, 0

    def predict(key, index=None):
        values = x[key] if index is None else x[key][index]
        if not learn_probe:
            mu = cached[key] if index is None else cached[key][index]
            return model.scores_from_moments(mu, theta)
        return model.scores(values, theta, probe, config['encoding'])

    def assess(epoch):
        nonlocal best, best_key, stale
        with torch.no_grad():
            vf = predict('validation')
            if not bool(torch.isfinite(vf).all()) or float(vf.abs().max()) > 1+1e-8:
                raise ArithmeticError('Invalid normalized expectation')
            pred = vf.cpu().numpy()
            metric = primary(config['dataset'], data['validation']['y'], pred)
            loss = float(probability_loss(vf, y['validation'], weights['validation']))
            row = dict(epoch=epoch, updates=updates, validation_metric=metric, validation_loss=loss,
                       learning_rate=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            key = (-metric, loss, epoch)
            if best_key is None or key < best_key:
                best_key = key
                best = dict(theta=theta.detach().cpu().numpy().copy(), probe=probe.detach().cpu().numpy().copy(),
                            prediction=pred.copy(), epoch=epoch, updates=updates, metric=metric, loss=loss)
                stale = 0
            else:
                stale += 1
            history.append(row)
            return loss

    def save_progress(epoch):
        # Each validation boundary has durable selected/current parameters and
        # full history. A retry intentionally restarts and archives this state.
        temp = out/'best.tmp.npz'
        np.savez_compressed(temp, **best)
        temp.replace(out/'best.npz')
        checkpoint = dict(epoch=epoch, updates=updates, theta=theta.detach().cpu(), probe=probe.detach().cpu(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          shuffle_state=shuffle.get_state(), stale=stale, best_key=best_key,
                          identity_sha256=completion_fields(out, identity, ())['identity_sha256'])
        torch.save(checkpoint, out/'current.pt.tmp')
        (out/'current.pt.tmp').replace(out/'current.pt')
        write_json(out/'history.json', history)
        write_json(out/'progress.json', dict(history[-1], best_epoch=best['epoch'], best_validation_metric=best['metric']))

    assess(0)
    save_progress(0)
    for epoch in range(1, config['epochs']+1):
        permutation = torch.randperm(len(y['fit']), generator=shuffle)
        for start in range(0, len(permutation), config['batch_size']):
            idx = permutation[start:start+config['batch_size']].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = probability_loss(predict('fit', idx), y['fit'][idx], weights['fit'][idx])
            if not bool(torch.isfinite(loss)):
                raise ArithmeticError('Nonfinite fitting loss')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
                raise ArithmeticError('Nonfinite or missing gradient')
            optimizer.step()
            updates += 1
        if epoch % config['evaluate_every'] == 0 or epoch == config['epochs']:
            validation_loss = assess(epoch)
            scheduler.step(validation_loss)
            save_progress(epoch)
            if epoch >= config['min_epochs'] and stale >= config['patience_evaluations']:
                break
    np.savez_compressed(out/'parameters.npz', theta=best['theta'], probe=best['probe'], initial_theta=theta0.numpy(),
                        initial_probe=probe0.numpy(), terminal_theta=theta.detach().cpu().numpy(), terminal_probe=probe.detach().cpu().numpy())
    np.savez_compressed(out/'validation.npz', ids=data['validation']['ids'], y=data['validation']['y'], scores=best['prediction'])
    write_json(out/'history.json', history)
    result = dict(**completion_fields(out, identity, ARTIFACTS), parameters_sha256=digest(out/'parameters.npz'), validation_sha256=digest(out/'validation.npz'),
                  best_epoch=best['epoch'], best_updates=best['updates'], validation_metric=best['metric'], validation_loss=best['loss'],
                  actual_epochs=epoch, actual_updates=updates, seconds=time.perf_counter()-started,
                  preparation_seconds=preparation_seconds, class_weights=cw.tolist(),
                  thresholds=validation_thresholds(data['validation']['y'], best['prediction']), probe_angles=best['probe'].tolist())
    write_json(out/'complete.json', result)
    return result


def calibration_tasks(protocol):
    tasks = []
    for dataset, split, encoding, slots, probe, init in itertools.product(
            ['magic','htru2'], range(5), ['iqp','pauli'], [4,8], ['zero','plus','ry'], range(2)):
        config = dict(protocol['training'], dataset=dataset, split=split, encoding=encoding, T=slots, probe=probe,
                      seed=21000+100*split+init, weighted=dataset=='htru2', stage='calibration', init=init)
        name = f'calibration/{dataset}/s{split}/{encoding}-T{slots}-{probe}-i{init}'
        tasks.append(dict(name=name, config=config))
    return tasks


def select_and_make_final(protocol, root, data_dir=None):
    root = Path(root)
    verified = {}
    for task in calibration_tasks(protocol):
        directory = root / task['name']
        expected = task['config']
        if json.loads((directory/'config.json').read_text()) != expected:
            raise ValueError('Calibration config differs from requested protocol: ' + task['name'])
        folder = data_dir
        if folder is None:
            try:
                folder = json.loads((directory/'provenance.json').read_text())['data_directory']
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError('Calibration lacks immutable input provenance: ' + task['name']) from exc
        loaded = load_split(folder, expected['dataset'], expected['split'])
        identity, _ = identity_for(folder, expected, loaded)
        verified[task['name']] = verify_complete(directory, identity, ARTIFACTS)
    tasks, selections = [], []
    for dataset, split in itertools.product(['magic','htru2'], range(5)):
        candidates = []
        for encoding, slots, probe in itertools.product(['iqp','pauli'], [4,8], ['zero','plus','ry']):
            names = [f'calibration/{dataset}/s{split}/{encoding}-T{slots}-{probe}-i{i}' for i in range(2)]
            runs = [verified[name] for name in names]
            candidates.append(dict(encoding=encoding, T=slots, probe=probe,
                validation_metric=float(np.mean([r['validation_metric'] for r in runs])),
                validation_loss=float(np.mean([r['validation_loss'] for r in runs])),
                probe_angles=min(enumerate(runs), key=lambda z:(-z[1]['validation_metric'],z[1]['validation_loss'],z[0]))[1]['probe_angles']))
        # Lower depth then simpler probe then encoder order break exact score/loss ties.
        selected = min(candidates, key=lambda r:(-r['validation_metric'],r['validation_loss'],r['T'],['zero','plus','ry'].index(r['probe']),['iqp','pauli'].index(r['encoding'])))
        selections.append(dict(dataset=dataset, split=split, selected=selected, candidates=candidates,
            calibration_complete_sha256={name:digest(root/name/'complete.json') for name in verified
                                         if name.startswith(f'calibration/{dataset}/s{split}/')}))
        for init in range(3):
            for weighted in ([True,False] if dataset=='htru2' else [False]):
                stage = 'final' if weighted == (dataset=='htru2') else 'unweighted'
                config = dict(protocol['training'],dataset=dataset,split=split,encoding=selected['encoding'],T=selected['T'],
                              probe=selected['probe'],fixed_probe=selected['probe_angles'],seed=31000+100*split+init,
                              weighted=weighted,stage=stage,init=init)
                tasks.append(dict(name=f'{stage}/{dataset}/s{split}/i{init}',config=config))
    write_json(root/'selections.json', selections)
    return tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    result = fit(args.data,args.out,json.loads(Path(args.config).read_text()))
    print(json.dumps(result))
