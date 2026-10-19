"""Declared finite classical comparators, validation selection and portable recipes."""
from __future__ import annotations
import itertools
import shutil
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from dataset_screen.evaluate import save_npz
from . import data,evidence,train,kernels as k
METHODS=('lr_standardized','lr_angles','rbf_svm','gradient_boosting')

def grid(d):
    p=train.protocol()['classical']
    return dict(lr_standardized=[dict(C=c) for c in p['lr_C']],lr_angles=[dict(C=c) for c in p['lr_C']],
                rbf_svm=[dict(C=c,gamma=g/d) for c in p['rbf_C'] for g in p['rbf_gamma_multipliers']],
                gradient_boosting=[dict(learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=l2) for lr,leaves,l2 in itertools.product(p['hgb_learning_rate'],p['hgb_leaves'],p['hgb_l2'])])

def fit(data_dir,out_dir,config,*,pin_path):
    train._verify_pin(data_dir,pin_path)
    if set(config)!={'dataset','resample'} or config['dataset'] not in train.protocol()['datasets'] or type(config['resample']) is not int or config['resample'] not in range(5): raise ValueError('Literal classical task required')
    split=data.load_split(data_dir,config['dataset'],config['resample'])
    def verify_source():
        train._verify_pin(data_dir,pin_path)
        if evidence.input_identity(data.load_split(data_dir,config['dataset'],config['resample']))!=evidence.input_identity(split):raise ValueError('Source/input changed during classical fitting')
    return _fit(split,out_dir,config,grid(len(split['mean'])),'formal',verify_source=verify_source)

def fit_synthetic(split,out_dir,config):
    return _fit(split,out_dir,config,{m:v[:2] for m,v in grid(len(split['mean'])).items()},'synthetic')

def _portable(estimator,method,fit_x):
    if method.startswith('lr_'): return dict(coef=estimator.coef_[0],intercept=estimator.intercept_)
    if method=='rbf_svm': return dict(support_indices=estimator.support_.astype(np.int64),support_x=fit_x[estimator.support_],dual_coef=estimator.dual_coef_[0],intercept=estimator.intercept_,gamma=np.array(estimator._gamma))
    # sklearn's raw tree leaf values already include shrinkage. Store numeric
    # node arrays and baseline only; no sklearn object is needed at inference.
    return dict(baseline=np.asarray(estimator._baseline_prediction).reshape(-1),**{f'tree_{i:03d}':v[0].nodes.copy() for i,v in enumerate(estimator._predictors)})

def _fit(split,out_dir,config,candidates,mode,verify_source=lambda:None):
    k.validate(split,mode);out=Path(out_dir);identity=k._identity(split,config,mode,'classical',candidates=candidates)
    existing=evidence.begin(out,identity)
    if existing is not None:return existing
    start,cpu=time.monotonic(),time.process_time();k.save_inputs(out,split);selected={};outcomes=[]
    y=split['fit']['y'];counts=np.bincount(y,minlength=2);weights=len(y)/(2*counts)
    with threadpool_limits(limits=1):
        for method in METHODS:
            rep='angles' if method=='lr_angles' else 'standardized';xf,xv=split['fit'][rep],split['validation'][rep];successful=[];models={}
            for index,candidate in enumerate(candidates[method]):
                cs,cc=time.monotonic(),time.process_time();item=dict(method=method,index=index,candidate=candidate)
                data.write_json(out/'progress.json',dict(stage='candidate',current=item,finished=len(outcomes)))
                try:
                    if method.startswith('lr_'): estimator=LogisticRegression(**candidate,solver='lbfgs',tol=1e-8,max_iter=5000,class_weight='balanced')
                    elif method=='rbf_svm': estimator=SVC(**candidate,kernel='rbf',class_weight='balanced',tol=.001,cache_size=512,max_iter=-1)
                    else: estimator=HistGradientBoostingClassifier(**candidate,max_iter=300,early_stopping=False,random_state=39011+config['resample'])
                    if method=='gradient_boosting': estimator.fit(xf,y,sample_weight=weights[y])
                    else: estimator.fit(xf,y)
                    scores=estimator.decision_function(xv);item.update(status='complete',metrics=k.metrics(split['validation']['y'],scores))
                    if hasattr(estimator,'n_iter_'):item['iterations']=np.asarray(estimator.n_iter_).tolist()
                    if hasattr(estimator,'support_'):item['support_count']=len(estimator.support_)
                    models[index]=_portable(estimator,method,split['fit']['x']);successful.append(item)
                    k.save_predictions(out/'candidates'/f'{method}_{index:03d}.npz',split['validation'],scores)
                except Exception as exc:item.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                item['costs']=k.costs(cs,cc);data.write_json(out/'candidates'/f'{method}_{index:03d}.json',item);outcomes.append(item)
            if successful:
                winner=min(successful,key=lambda v:k.selection_key(v,config['dataset']));selected[method]=winner
                save_npz(out/'selected'/f'{method}.npz',**models[winner['index']]);shutil.copyfile(out/'candidates'/f"{method}_{winner['index']:03d}.npz",out/('validation_'+method+'.npz'))
    if any(v['status']!='complete' for v in outcomes): raise ValueError('Failed candidates preserved; bundle unsealed')
    verify_source()
    if evidence.input_identity(split)!=identity['inputs']:raise ValueError('Input mutated during fitting')
    return k._finish(out,identity,dict(kind='classical',selected=selected,candidates=candidates),dict(selected=selected,candidate_count=len(outcomes),costs=k.costs(start,cpu)))

def tree_scores(nodes,x):
    result=np.empty(len(x),dtype=np.float64)
    for row,features in enumerate(x):
        index=0
        while not nodes[index]['is_leaf']:
            node=nodes[index]
            if node['is_categorical']:raise ValueError('Categorical tree unsupported by declared numeric protocol')
            value=features[int(node['feature_idx'])]
            left=bool(node['missing_go_to_left']) if np.isnan(value) else value<=node['num_threshold']
            index=int(node['left'] if left else node['right'])
        result[row]=nodes[index]['value']
    return result

def predict_original_features(out_dir,x,method,*,evidence_mode='formal'):
    out=Path(out_dir);evidence.authenticate(out,evidence_mode=evidence_mode);recipe=k.read(out/'recipe.json')
    if method not in recipe['selected']:raise ValueError('Unknown selected classical model')
    pre=k.arrays(out/'preprocessing.npz');params=k.arrays(out/'selected'/f'{method}.npz');x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(pre['mean']) or not np.isfinite(x).all():raise ValueError('Invalid raw features')
    z=(x-pre['mean'])/pre['scale']
    if method.startswith('lr_'):
        rep=.5+np.arctan(z)/np.pi if method=='lr_angles' else z
        return rep@params['coef']+params['intercept'][0]
    if method=='rbf_svm':
        support=(params['support_x']-pre['mean'])/pre['scale'];gamma=float(params['gamma'])
        # Direct differences avoid cancellation and need no production kernel.
        result=np.empty(len(z))
        for start in range(0,len(z),256):
            distances=sum((z[start:start+256,j,None]-support[None,:,j])**2 for j in range(z.shape[1]))
            result[start:start+256]=np.exp(-gamma*distances)@params['dual_coef']+params['intercept'][0]
        return result
    scores=np.full(len(z),params['baseline'][0],dtype=np.float64)
    for name,nodes in sorted(params.items()):
        if name.startswith('tree_'):scores+=tree_scores(nodes,z)
    return scores
