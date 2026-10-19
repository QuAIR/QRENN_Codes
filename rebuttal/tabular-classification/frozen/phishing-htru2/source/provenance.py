"""Fail-closed fit identities and durable, separately identified retry attempts."""
from pathlib import Path
import hashlib
import json
import time
import uuid

import numpy as np
import sklearn

from data import digest, write_json


def config_digest(config):
    # Preserve the pre-existing COMPLETE configuration identity convention.
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_identity(data_dir, dataset, split, config, source_files, class_weights):
    folder = Path(data_dir)
    manifest = json.loads((folder / 'manifest.json').read_text())
    raw_hash = digest(folder / (dataset + '.npz'))
    split_hash = digest(folder / f'{dataset}-split-{split}.npz')
    matches = [s for s in manifest['splits'] if s['dataset'] == dataset and s['split'] == split]
    if (manifest['datasets'][dataset]['parsed_sha256'] != raw_hash or len(matches) != 1
            or matches[0]['archive_sha256'] != split_hash):
        raise ValueError('Data bytes do not match the immutable manifest')
    source_root = Path(__file__).resolve().parent
    return dict(schema='a07-fit-identity-v1', config=config, config_sha256=config_digest(config),
        sources={name: digest(source_root / name) for name in source_files},
        data=dict(dataset=dataset, split=split, parsed_sha256=raw_hash, split_sha256=split_hash,
                  manifest_sha256=digest(folder / 'manifest.json')),
        class_weights=np.asarray(class_weights).tolist(),
        runtime=dict(numpy=np.__version__, sklearn=sklearn.__version__))


def verify_complete(out, identity, artifacts):
    out = Path(out)
    try:
        complete = json.loads((out / 'complete.json').read_text())
        provenance = json.loads((out / 'provenance.json').read_text())
        if (json.loads((out / 'config.json').read_text()) != identity['config']
                or provenance['identity'] != identity
                or complete['config_sha256'] != identity['config_sha256']
                or complete['identity_sha256'] != config_digest(identity)):
            raise ValueError('Existing run has different source, data or protocol identity')
        for name in artifacts:
            if digest(out / name) != complete['artifacts_sha256'][name]:
                raise ValueError('Completed artifact checksum mismatch: ' + name)
        # Keep and check established public fields as well as the full map.
        for name, field in [('parameters.npz', 'parameters_sha256'), ('model.joblib', 'model_sha256'),
                            ('validation.npz', 'validation_sha256')]:
            if name in artifacts and digest(out / name) != complete[field]:
                raise ValueError('Completed artifact checksum mismatch: ' + name)
        return complete
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError('Missing or invalid provenance/completed artifacts: ' + str(out)) from exc


def begin_attempt(out, identity, data_dir):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    active = [p for p in out.iterdir() if p.name != 'attempts']
    previous = None
    if active:
        try:
            old_config = json.loads((out / 'config.json').read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('Incomplete attempt has no valid config; preserve and inspect it') from exc
        if old_config != identity['config']:
            raise ValueError('Refusing to overwrite a different incomplete protocol')
        if (out / 'provenance.json').exists():
            if json.loads((out / 'provenance.json').read_text())['identity'] != identity:
                raise ValueError('Incomplete attempt has different source or data identity')
        # Legacy interrupted attempts are preserved in full before deterministic restart.
        archive = out / 'attempts' / (str(time.time_ns()) + '-' + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        for path in active:
            path.replace(archive / path.name)
        previous = str(archive.relative_to(out))
    write_json(out / 'config.json', identity['config'])
    write_json(out / 'provenance.json', dict(identity=identity, data_directory=str(Path(data_dir).resolve())))
    write_json(out / 'attempt.json', dict(attempt_id=uuid.uuid4().hex, started_at=time.time(),
        mode='deterministic_restart' if previous else 'fresh', previous_attempt=previous,
        resume_policy='restart from recorded seed; preserve prior trajectory and checkpoints'))


def completion_fields(out, identity, artifacts):
    return dict(config_sha256=identity['config_sha256'], identity_sha256=config_digest(identity),
                artifacts_sha256={name: digest(Path(out) / name) for name in artifacts})


def attempt_id(out):
    path = Path(out) / 'attempt.json'
    return json.loads(path.read_text())['attempt_id'] if path.exists() else None


def record_failure(out, previous_id, error):
    current = attempt_id(out)
    if current is not None and current != previous_id and not (Path(out)/'complete.json').exists():
        write_json(Path(out)/'failure.json', dict(attempt_id=current, failed_at=time.time(),
            error_type=type(error).__name__, message=str(error)))
