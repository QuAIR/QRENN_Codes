"""Verify the reviewed release, prepare official data and freeze execution."""
from pathlib import Path
import hashlib,json,tarfile,os,sys
root=Path(sys.argv[1]).resolve()
record=json.loads((root/'runtime/deployment.json').read_text())
archive=root/'runtime/source.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['archive_sha256']
source=root/record['release']
assert not source.exists(), 'Existing release preserved'
source.mkdir()
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        p=Path(member.name)
        assert not p.is_absolute() and '..' not in p.parts and member.isfile()
    tar.extractall(source)
for name,sha in record['files'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
os.chdir(source);sys.path.insert(0,str(source))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from multidata_matched import data,train,evidence
assert evidence.source_hashes()==record['production']
manifest=data.prepare(root/'raw',root/'data')
assert data.digest(root/'data/manifest.json')==record['manifest_sha256']
train.create_pin(root/'data',root/'pin.json')
assert data.digest(root/'pin.json')==record['pin_sha256']
import numpy as np,torch
from dataset_screen.model import initial_parameters,scores
from eigenbasis_study import model
from multidata_matched.evaluate import numpy_states
smoke=[]
for dataset in ('shoppers','rice'):
    split=data.load_split(root/'data',dataset,0)
    x=split['validation']['x'][:3];a=torch.tensor(split['validation']['angles'][:3],dtype=torch.float64)
    d=a.shape[1];n=(d-1).bit_length()
    theta,probe=initial_parameters(d,'diagonal_joint',8,82000)
    theta.requires_grad_();probe.requires_grad_()
    predicted=model.scores_prepared(model.prepare_inputs(a,'fixed',n=n),theta,probe)
    reference=numpy_states(x,split['mean'],split['scale'],theta.detach().numpy(),probe.detach().numpy())
    direct=(abs(reference.reshape(len(x),4,-1))**2).sum(-1)@np.array([1.,-1.,-1.,1.])
    error=float(np.max(np.abs(predicted.detach().numpy()-direct)));assert error<2e-12
    grads=torch.autograd.grad(predicted.sum(),(theta,probe));assert all(torch.isfinite(g).all() for g in grads)
    smoke.append(dict(dataset=dataset,features=d,n=n,initial_score_error=error))
result=dict(source=record['release'],source_commit=record['source_commit'],source_files=len(record['production']),splits=len(manifest['splits']),manifest_sha256=data.digest(root/'data/manifest.json'),source_archive_sha256=data.digest(archive),pin_sha256=data.digest(root/'pin.json'),status='prepared',dimension_checks=smoke)
data.write_json(root/'runtime/server-preparation.json',result)
print(json.dumps(result),flush=True)
