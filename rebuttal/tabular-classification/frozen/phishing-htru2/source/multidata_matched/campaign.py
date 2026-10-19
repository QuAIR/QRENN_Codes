"""Durable bounded development orchestration; test access is always explicit.

A campaign never automatically freezes or evaluates. Run-development may be
repeated to authenticate completed jobs and retry failures into fresh attempts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import time
import uuid
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
from . import data,evidence,train,kernels,baselines,evaluate


def tasks():
    result=[]
    for dataset in train.protocol()['datasets']:
        for r in range(5):
            result.append(dict(kind='classical',config=dict(dataset=dataset,resample=r)))
            for i in range(3):
                result.append(dict(kind='qrenn',config=dict(dataset=dataset,resample=r,init=i)))
                for arm in ('initial','trained'):result.append(dict(kind='kernel',config=dict(dataset=dataset,resample=r,init=i,arm=arm)))
    return result

def task_name(task):
    c=task['config'];return f"{c['dataset']}_r{c['resample']}_{task['kind']}"+('' if task['kind']=='classical' else f"_i{c['init']}"+('' if task['kind']=='qrenn' else '_'+c['arm']))

def create(data_dir,pin_path,out_dir,*,qrenn_root,phishing_kernel_root,legacy_data,legacy_qrenn):
    train._verify_pin(data_dir,pin_path);out=Path(out_dir)
    if out.exists():raise ValueError('Existing campaign preserved')
    out.mkdir(parents=True)
    value=dict(schema='qrenn-a07-multidata-campaign-v1',source_hashes=evidence.source_hashes(),pin_sha256=data.digest(pin_path),manifest_sha256=data.digest(Path(data_dir)/'manifest.json'),
               paths={key:str(Path(value).resolve()) for key,value in dict(data=data_dir,pin=pin_path,qrenn=qrenn_root,phishing_kernel=phishing_kernel_root,legacy_data=legacy_data,legacy_qrenn=legacy_qrenn).items()},
               tasks={task_name(t):dict(t,status='pending',attempts=[]) for t in tasks()},test_access='controller explicit freeze and evaluate only')
    data.write_json(out/'campaign.json',value);return value

def _verify(out):
    value=kernels.read(Path(out)/'campaign.json');paths=value['paths'];train._verify_pin(paths['data'],paths['pin'])
    if value['source_hashes']!=evidence.source_hashes() or value['pin_sha256']!=data.digest(paths['pin']) or value['manifest_sha256']!=data.digest(Path(paths['data'])/'manifest.json'):raise ValueError('Campaign source/data changed')
    if set(value['tasks'])!={task_name(t) for t in tasks()}:raise ValueError('Campaign task inventory differs')
    for t in tasks():
        retained=value['tasks'][task_name(t)]
        if any(retained[key]!=t[key] for key in ('kind','config')):raise ValueError('Campaign task identity differs')
    return value

def _execute(paths,task,out):
    import torch
    torch.set_num_threads(1)
    c=task['config'];checkpoint=Path(paths['qrenn'])/'results'/c['dataset']/f"r{c['resample']}"/f"i{c.get('init',0)}"
    if task['kind']=='classical':return baselines.fit(paths['data'],out,c,pin_path=paths['pin'])
    if task['kind']=='qrenn':return kernels.import_qrenn(paths['data'],out,checkpoint,c,pin_path=paths['pin'],legacy_data=paths['legacy_data'],legacy_qrenn=paths['legacy_qrenn'])
    if c['dataset']=='phishing':
        source=Path(paths['phishing_kernel'])/'results'/c['arm']/f"r{c['resample']}"/f"i{c['init']}"
        return kernels.import_phishing(paths['data'],out,source,c,legacy_data=paths['legacy_data'],legacy_qrenn=paths['legacy_qrenn'])
    return kernels.fit(paths['data'],out,checkpoint,c,pin_path=paths['pin'])

def run_development(out_dir,*,workers=1,limit=None,recover_stale_lock=False):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('Comparison worker bound is 1..8')
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('Positive task limit required')
    out=Path(out_dir);lock=out/'controller.lock'
    if recover_stale_lock and lock.exists():
        pid=int(lock.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:lock.unlink()
        else:raise ValueError('Controller PID still live; do not interrupt it')
    # Exclusive creation rejects concurrent controllers without changing work.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        value=_verify(out);pending=[]
        for name,task in value['tasks'].items():
            if task['status']=='complete':
                completed=task['attempts'][-1];evidence.authenticate(completed['path'],expected_config=task['config'])
                if data.digest(Path(completed['path'])/'completion.json')!=completed['completion_sha256']:raise ValueError('Completed task changed')
            else:pending.append((name,task))
        if limit is not None:pending=pending[:limit]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={}
            for name,task in pending:
                attempt=dict(path=str((out/'attempts'/name/uuid.uuid4().hex).resolve()),status='running',started_unix=time.time())
                task['attempts'].append(attempt);task['status']='running';data.write_json(out/'campaign.json',value)
                futures[pool.submit(_execute,value['paths'],task,attempt['path'])]=(name,task,attempt)
            for future in as_completed(futures):
                name,task,attempt=futures[future]
                try:
                    future.result();evidence.authenticate(attempt['path'],expected_config=task['config']);attempt.update(status='complete',completion_sha256=data.digest(Path(attempt['path'])/'completion.json'));task['status']='complete'
                except Exception as exc:attempt.update(status='failed',error_type=type(exc).__name__,error=str(exc));task['status']='failed'
                attempt['finished_unix']=time.time();data.write_json(out/'campaign.json',value)
        return value
    finally:lock.unlink()

def freeze(out_dir,snapshot):
    value=_verify(out_dir)
    if any(t['status']!='complete' for t in value['tasks'].values()):raise ValueError('Development campaign incomplete')
    bundles=[t['attempts'][-1]['path'] for t in value['tasks'].values()]
    return evaluate.freeze(value['paths']['data'],bundles,snapshot,pin_path=value['paths']['pin'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    create_parser=sub.add_parser('create')
    for key in ('data','pin','out','qrenn-root','phishing-kernel-root','legacy-data','legacy-qrenn'):create_parser.add_argument('--'+key,required=True)
    run=sub.add_parser('run-development');run.add_argument('--out',required=True);run.add_argument('--workers',type=int,default=1);run.add_argument('--limit',type=int);run.add_argument('--recover-stale-lock',action='store_true')
    pin=sub.add_parser('freeze');pin.add_argument('--out',required=True);pin.add_argument('--snapshot',required=True)
    test=sub.add_parser('evaluate');test.add_argument('--data',required=True);test.add_argument('--snapshot',required=True);test.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.operation=='create':result=create(args.data,args.pin,args.out,qrenn_root=args.qrenn_root,phishing_kernel_root=args.phishing_kernel_root,legacy_data=args.legacy_data,legacy_qrenn=args.legacy_qrenn)
    elif args.operation=='run-development':result=run_development(args.out,workers=args.workers,limit=args.limit,recover_stale_lock=args.recover_stale_lock)
    elif args.operation=='freeze':result=freeze(args.out,args.snapshot)
    else:result=evaluate.evaluate(args.data,args.snapshot,args.out)
    print(json.dumps(dict(status='complete',operation=args.operation)))
if __name__=='__main__':main()
