"""Download, check and replay the three published tabular datasets."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
RELEASE = json.loads((HERE / 'release.json').read_text())
CAMPAIGNS = {'phishing': 'phishing-htru2', 'htru2': 'phishing-htru2', 'rice': 'rice'}

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / RELEASE['asset']
    if not archive.exists():
        def download(record):
            path = destination/record['name']
            if not path.exists():
                partial = path.with_suffix(path.suffix+'.download')
                urllib.request.urlretrieve(record['url'], partial)
                if digest(partial) != record['sha256']:
                    raise ValueError('Downloaded part checksum differs: '+record['name'])
                partial.rename(path)
            if digest(path) != record['sha256']:
                raise ValueError('Part checksum differs: '+record['name'])
            return path
        if 'parts' in RELEASE:
            with ThreadPoolExecutor(max_workers=8) as pool:
                parts = list(pool.map(download, RELEASE['parts']))
            partial = archive.with_suffix(archive.suffix+'.assembling')
            with partial.open('wb') as f:
                for part in parts:
                    with part.open('rb') as source: shutil.copyfileobj(source, f)
            if digest(partial) != RELEASE['sha256']:
                raise ValueError('Assembled archive checksum differs')
            partial.rename(archive)
        else:
            download(dict(name=RELEASE['asset'], url=RELEASE['url'], sha256=RELEASE['sha256']))
    if digest(archive) != RELEASE['sha256']:
        raise ValueError('Archive checksum differs')
    if (destination / 'tabular-data').exists():
        raise ValueError('Extraction destination exists; use verify or a new directory')
    with tarfile.open(archive) as t:
        regular_files = set()
        for m in t.getmembers():
            p = Path(m.name)
            if p.is_absolute() or '..' in p.parts or not (m.isdir() or m.isfile() or m.islnk()):
                raise ValueError('Unexpected archive path or type')
            if m.islnk() and m.linkname not in regular_files:
                raise ValueError('Archive hard link must target a preceding regular file')
            if m.isfile(): regular_files.add(m.name)
        t.extractall(destination, filter='data')
    print(destination / 'tabular-data')

def verify_bytes(root):
    if digest(root/'SHA256.json') != RELEASE['payload_manifest_sha256']:
        raise ValueError('Payload manifest differs from the published release')
    manifest = json.loads((root / 'SHA256.json').read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'SHA256.json'}:
        raise ValueError('Payload file inventory differs')
    for name, record in manifest.items():
        p = root / name
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError('Payload checksum differs: ' + name)
    print(json.dumps({'payload_files_verified': len(manifest)}), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('download'); f.add_argument('--out', type=Path, required=True)
    for command in ['verify', 'replay', 'train']:
        s = sub.add_parser(command); s.add_argument('--data', type=Path, required=True)
        s.add_argument('--dataset', choices=CAMPAIGNS, required=command == 'train')
        if command == 'replay': s.add_argument('--all', action='store_true', help='Replay every selected model; default is resample 0 and initialization 0')
        if command == 'train':
            s.add_argument('--resample', type=int, choices=range(5), default=0)
            s.add_argument('--init', type=int, choices=range(3), default=0)
            s.add_argument('--out', type=Path, required=True)
            s.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.command == 'download': return fetch(a.out.resolve())
    root = a.data.resolve(); verify_bytes(root)
    tags = [CAMPAIGNS[a.dataset]] if a.dataset else list(dict.fromkeys(CAMPAIGNS.values()))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for tag in tags:
        args = [sys.executable, '-B', str(HERE/'worker.py'), a.command, '--campaign', tag, '--data', str(root)]
        if a.dataset: args += ['--dataset', a.dataset]
        if a.command == 'replay' and a.all: args += ['--all']
        if a.command == 'train':
            args += ['--resample', str(a.resample), '--init', str(a.init), '--out', str(a.out.resolve())]
            if a.dry_run: args += ['--dry-run']
        subprocess.run(args, check=True, env=env)

if __name__ == '__main__': main()
