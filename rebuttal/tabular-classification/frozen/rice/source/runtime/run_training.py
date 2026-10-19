"""Run all frozen paired development jobs; no dataset selection or test access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    assert 1<=args.workers<=24
    root=args.root.resolve();source=args.source.resolve();data=args.data.resolve()
    sys.path.insert(0,str(source))
    from multidata_matched.train import make_tasks,_verify_pin
    from multidata_matched.data import write_json
    from multidata_matched.evidence import authenticate
    pin=root/'pin.json';_verify_pin(data,pin)
    tasks=make_tasks();runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    state_path=runtime/'campaign-state.json'
    if state_path.exists():raise ValueError('Existing campaign preserved; no silent restart')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONPATH=str(source),PYTHONUNBUFFERED='1')
    state=dict(status='running',started=time.time(),workers=args.workers,
        source=str(source),data=str(data),pin_sha256=hashlib.sha256(pin.read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jobs={})
    pending=[];active={}
    for config in tasks:
        name=f"{config['dataset']}/r{config['resample']}/i{config['init']}"
        assert not (root/'results'/name).exists(), 'Never overwrite an existing scientific attempt'
        pending.append((name,config));state['jobs'][name]=dict(status='queued',config=config)
    def save():write_json(state_path,state)
    def interrupted(signum,frame):
        state.update(status='interrupted',signal=signum,finished=time.time());save()
        for process in active.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    while pending or active:
        while pending and len(active)<args.workers:
            name,config=pending.pop(0)
            command=[sys.executable,'-m','multidata_matched.train','--dataset',config['dataset'],
                '--r',str(config['resample']),'--i',str(config['init']),'--data',str(data),
                '--pin',str(pin),'--out',str(root/'results'/name)]
            log_path=runtime/(name.replace('/','__')+'.log')
            with log_path.open('xb') as stream:
                process=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=process
            state['jobs'][name].update(status='running',pid=process.pid,started=time.time(),command=command)
            save()
        for name,process in list(active.items()):
            result=process.poll()
            if result is not None:
                if result==0:
                    try: authenticate(root/'results'/name)
                    except Exception as exc:
                        result=2
                        state['jobs'][name]['authentication_error']=str(exc)
                state['jobs'][name].update(status='complete' if result==0 else 'failed',returncode=result,finished=time.time())
                del active[name];save()
        state['heartbeat']=time.time();save()
        if pending or active:time.sleep(5)
    good=all(v['status']=='complete' for v in state['jobs'].values())
    state.update(status='complete' if good else 'failed',finished=time.time());save()
    return 0 if good else 1


if __name__=='__main__':raise SystemExit(main())
