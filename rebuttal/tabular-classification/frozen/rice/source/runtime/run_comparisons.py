"""Bounded development, explicit model freeze, and one held-out evaluation."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

def main():
    root=Path(sys.argv[1]).resolve()
    deployment=json.loads((root/'runtime/deployment.json').read_text())
    source=root/deployment['release']
    sys.path.insert(0,str(source));os.chdir(source)
    from multidata_matched import data,train,campaign,evidence,evaluation_parallel
    status=root/'runtime/comparison-controller.json'
    if status.exists(): raise ValueError('Existing controller preserved')
    state=dict(stage='waiting_for_training',started=time.time(),pid=os.getpid(),controller_sha256=data.digest(__file__),source_commit=deployment['source_commit'])
    def save():data.write_json(status,state)
    save()
    try:
        while True:
            p=root/'runtime/campaign-state.json'
            if p.exists():
                training=json.loads(p.read_text())
                if training['status']=='complete': break
                if training['status'] in ('failed','interrupted'):raise ValueError('Training failed; preserve attempts before recovery')
            time.sleep(10)
        jobs=training['jobs'];assert len(jobs)==30 and all(j['status']=='complete' for j in jobs.values())
        train._verify_pin(root/'data',root/'pin.json')
        for name in jobs:evidence.authenticate(root/'results'/name)
        state.update(stage='development_comparisons',training_completed=time.time());save()
        campaign.create(root/'data',root/'pin.json',root/'comparisons',qrenn_root=root,
            phishing_kernel_root=root/'unused',legacy_data=root/'unused',legacy_qrenn=root/'unused')
        value=campaign.run_development(root/'comparisons',workers=8)
        assert len(value['tasks'])==100 and all(t['status']=='complete' for t in value['tasks'].values()),'Comparison failure preserved'
        state.update(stage='freezing_selected_models');save()
        frozen=campaign.freeze(root/'comparisons',root/'freeze.json')
        assert frozen['selected_count']==190
        state.update(stage='explicit_test_evaluation',snapshot_sha256=frozen['snapshot_sha256'],test_evaluation_started=time.time());save()
        result=evaluation_parallel.evaluate(root/'data',root/'freeze.json',root/'evaluation',workers=8)
        assert result['selected_model_count']==190
        state.update(stage='evaluation_complete',evaluation_finished=time.time(),selected_models=190);save()
        subprocess.run([sys.executable,str(root/'runtime/seal_delivery.py'),str(root)],check=True)
        state.update(stage='complete',finished=time.time());save()
    except Exception as exc:
        state.update(failed_stage=state['stage'],stage='failed',error_type=type(exc).__name__,error=str(exc),finished=time.time());save()
        raise
if __name__=='__main__':main()
