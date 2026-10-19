"""Seal authenticated scientific records; transport/authentication files excluded."""
from pathlib import Path
import sys,json,tarfile
root=Path(sys.argv[1]).resolve()
deployment=json.loads((root/'runtime/deployment.json').read_text())
source=root/deployment['release'];sys.path.insert(0,str(source))
from multidata_matched import data,evidence,evaluate
assert json.loads((root/'runtime/comparison-controller.json').read_text())['stage']=='evaluation_complete'
frozen=evaluate.authenticate_snapshot(root/'freeze.json')
completion=json.loads((root/'evaluation/completion.json').read_text())
assert completion['artifacts']==evidence.inventory(root/'evaluation',exclude=('completion.json',))
assert completion['selected_model_count']==190
files={}
def add(path,name):
    path=Path(path);p=Path(name)
    assert path.is_file() and not path.is_symlink()
    assert name not in files and not p.is_absolute() and '..' not in p.parts
    files[name]=path
def tree(path,prefix):
    assert Path(path).is_dir(),path
    for f in sorted(Path(path).rglob('*')):
        assert not f.is_symlink()
        if f.is_file():add(f,str(Path(prefix)/f.relative_to(path)))
for name,sha in sorted(deployment['files'].items()):
    assert data.digest(source/name)==sha
    add(source/name,'source/'+name)
for directory in ('data','raw','results','evaluation'):tree(root/directory,directory)
for record in frozen['bundles']:
    p=Path(record['path']);tree(p,'bundles/'+p.name)
for name in ('pin.json','freeze.json','comparisons/campaign.json'):add(root/name,name)
for name in ('deployment.json','server-preparation.json','campaign-state.json','comparison-controller.json','run_training.py','run_comparisons.py','seal_delivery.py'):
    add(root/'runtime'/name,'runtime/'+name)
manifest={name:dict(bytes=p.stat().st_size,sha256=data.digest(p)) for name,p in sorted(files.items())}
delivery=root/'delivery';delivery.mkdir(exist_ok=True)
assert not (delivery/'native-evidence.tar.gz').exists()
data.write_json(delivery/'native-manifest.json',dict(schema='qrenn-a07-shoppers-rice-native-v1',files=manifest,source_commit=deployment['source_commit'],selected_models=190,training_runs_new=30,privacy='Scientific records and code only; no credentials or conversations'))
with tarfile.open(delivery/'native-evidence.tar.gz','w:gz') as tar:
    for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
    tar.add(delivery/'native-manifest.json',arcname='native-manifest.json')
result=dict(archive='native-evidence.tar.gz',archive_bytes=(delivery/'native-evidence.tar.gz').stat().st_size,archive_sha256=data.digest(delivery/'native-evidence.tar.gz'),manifest_sha256=data.digest(delivery/'native-manifest.json'),files=len(files),uncompressed_bytes=sum(x['bytes'] for x in manifest.values()),selected_models=190)
data.write_json(delivery/'delivery.json',result);print(json.dumps(result,indent=2))
