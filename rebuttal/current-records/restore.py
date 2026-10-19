"""Restore archived records using byte checks only; never execute scientific code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
RELEASE = json.loads((HERE / 'release.json').read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    manifest_path = root / 'RECORDS.json'
    if digest(manifest_path) != RELEASE['manifest_sha256']:
        raise ValueError('Restored manifest checksum differs')
    manifest = json.loads(manifest_path.read_text())
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Unexpected symbolic link: ' + str(path))
        if path.is_file():
            actual.add(str(path.relative_to(root)))
    if actual != set(manifest) | {'RECORDS.json'}:
        raise ValueError('Restored file inventory differs')
    checked = {}
    for name, record in manifest.items():
        path = root / name
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in checked:
            checked[identity] = digest(path)
        if stat.st_size != record['bytes'] or checked[identity] != record['sha256']:
            raise ValueError('Restored file checksum differs: ' + name)
    result = {'payload_files_verified': len(manifest),
              'logical_bytes': sum(r['bytes'] for r in manifest.values()),
              'unique_inodes_hashed': len(checked), 'scientific_code_executed': False}
    print(json.dumps(result), flush=True)
    return result


def download(destination, workers):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / RELEASE['asset']
    root = destination / 'workspace'
    if root.exists():
        raise ValueError('Extraction destination exists; use verify or a new directory')

    def get_part(record):
        path = destination / record['name']
        if path.exists():
            if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise ValueError('Existing part checksum differs: ' + path.name)
            return path
        temporary = path.with_suffix(path.suffix + '.download')
        for attempt in range(4):
            try:
                request = urllib.request.Request(record['url'], headers={'User-Agent': 'QRENN-record-restoration'})
                with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
                    raise ValueError('Downloaded part checksum differs: ' + path.name)
                temporary.rename(path)
                print(json.dumps({'part_verified': path.name, 'bytes': record['bytes']}), flush=True)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))

    if not archive.exists():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(get_part, record) for record in RELEASE['parts']]
            for job in as_completed(jobs):
                job.result()
        temporary = archive.with_suffix(archive.suffix + '.assembling')
        with temporary.open('wb') as target:
            for record in RELEASE['parts']:
                with (destination / record['name']).open('rb') as source:
                    shutil.copyfileobj(source, target, 1024 * 1024)
        if temporary.stat().st_size != RELEASE['bytes'] or digest(temporary) != RELEASE['sha256']:
            raise ValueError('Assembled archive checksum differs')
        temporary.rename(archive)
    if archive.stat().st_size != RELEASE['bytes'] or digest(archive) != RELEASE['sha256']:
        raise ValueError('Archive checksum differs')
    print(json.dumps({'archive_verified': archive.name, 'bytes': RELEASE['bytes']}), flush=True)
    # Validate the entire member list before any extraction.
    with tarfile.open(archive) as bundle:
        names, regular = set(), set()
        for member in bundle.getmembers():
            path = Path(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts or
                    path.parts[0] != 'workspace' or member.name in names or
                    not (member.isdir() or member.isfile() or member.islnk())):
                raise ValueError('Unsafe or duplicate archive member: ' + member.name)
            if member.islnk() and member.linkname not in regular:
                raise ValueError('Hard link must target a preceding regular file')
            names.add(member.name)
            if member.isfile():
                regular.add(member.name)
        bundle.extractall(destination, filter='data')
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--out', type=Path, required=True)
    fetch.add_argument('--workers', type=int, default=8)
    check = commands.add_parser('verify')
    check.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'download':
        if not 1 <= args.workers <= 32:
            parser.error('workers must be between 1 and 32')
        download(args.out.resolve(), args.workers)
    else:
        verify(args.data.resolve())


if __name__ == '__main__':
    main()
