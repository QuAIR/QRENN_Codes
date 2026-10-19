"""Reversible official parsing and fit-only categorical feature expansion."""
from __future__ import annotations
import csv
import io
import json
import numpy as np
from scipy.io import arff

DATASETS = {
    'shoppers': dict(archive='468.zip', member='online_shoppers_intention.csv',
        archive_members=['online_shoppers_intention.csv'],
        url='https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
        sha256='2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14',
        rows=12330, features=17, positive=1908, groups=12205, conflicting_groups=0, conflicting_rows=0),
    'rice': dict(archive='545.zip', member='Rice_Cammeo_Osmancik.arff',
        archive_members=['Citation_Request.txt','Rice_Cammeo_Osmancik.arff'],
        url='https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip',
        sha256='fe94e42046b829de21b92b0ffb6a22774fde021328cae799faa802a14e8dbed9',
        rows=3810, features=7, positive=1630, groups=3810, conflicting_groups=0, conflicting_rows=0),
}


def parse(content, name, spec):
    text=content.decode('utf-8-sig')
    if name=='shoppers':
        reader=csv.reader(io.StringIO(text));fields=next(reader);tokens=np.asarray(list(reader),dtype=str)
        raw=tokens[:,:-1];labels=tokens[:,-1]
        dictionary={fields[j]:sorted(set(raw[:,j])) for j in range(10,17)}
        x=np.empty(raw.shape,dtype=np.float64);x[:,:10]=raw[:,:10].astype(np.float64)
        for j in range(10,17):
            # These IDs label tokens reversibly. Only equality indicators reach models.
            x[:,j]=np.searchsorted(dictionary[fields[j]],raw[:,j])
        if not np.isin(labels,['TRUE','FALSE']).all():raise ValueError('Unexpected Revenue labels')
        y=(labels=='TRUE').astype(np.int64)
    elif name=='rice':
        records,metadata=arff.loadarff(io.StringIO(text));fields=list(records.dtype.names)
        columns=[np.asarray(records[f],dtype=np.float64) for f in fields[:-1]]
        x=np.column_stack(columns);labels=np.char.decode(records[fields[-1]],'utf-8')
        # Retain numeric lexical tokens from the released data section as well.
        raw=np.asarray([next(csv.reader([line]))[:-1] for line in text.split('@DATA',1)[-1].splitlines()
                        if line.strip() and not line.lstrip().startswith('%')],dtype=str)
        if not np.isin(labels,['Cammeo','Osmancik']).all():raise ValueError('Unexpected Rice labels')
        y=(labels=='Cammeo').astype(np.int64);dictionary={}
    else:raise ValueError('Unknown additional dataset')
    if x.shape!=(spec['rows'],spec['features']) or raw.shape!=x.shape or not np.isfinite(x).all() or int(y.sum())!=spec['positive']:
        raise ValueError('Unexpected official shape/values/positive count: '+name)
    artifacts=dict(raw_tokens=raw,original_labels=labels,feature_names=np.asarray(fields[:-1]),
        token_dictionary_json=np.asarray(json.dumps(dictionary,sort_keys=True)))
    return np.ascontiguousarray(x),y,artifacts,dictionary


def fit_categories(x, fit_ids):
    return {j:np.unique(x[fit_ids,j]) for j in range(10,17)}


def transform_categories(x,categories):
    return np.column_stack([x[:,:10]]+[(x[:,j,None]==categories[j][None,:]).astype(np.float64) for j in range(10,17)])


def manual_reconstruction(raw, part, dataset):
    """Separate raw-token audit; deliberately does not call the transform helper."""
    tokens=raw['raw_tokens'];parsed=raw['x'];columns=[]
    if dataset=='shoppers':
        dictionary=json.loads(str(raw['token_dictionary_json']));fields=raw['feature_names']
        numeric=tokens[:,:10].astype(np.float64)
        if not np.array_equal(numeric,parsed[:,:10]):raise ValueError('Raw numeric parsing differs')
        columns.extend(numeric[:,j] for j in range(10))
        for j in range(10,17):
            all_tokens=dictionary[str(fields[j])]
            if [all_tokens[int(v)] for v in parsed[:,j]]!=tokens[:,j].tolist():raise ValueError('Raw token parsing differs')
            vocab=part[f'category_{j}'];fit_tokens=set(tokens[part['fit_ids'],j])
            if {all_tokens[int(v)] for v in vocab}!=fit_tokens:raise ValueError('Vocabulary not fit-only')
            for value in vocab:columns.append((tokens[:,j]==all_tokens[int(value)]).astype(np.float64))
    else:
        if not np.array_equal(tokens.astype(np.float64),parsed):raise ValueError('Raw numeric parsing differs')
        columns.extend(parsed[:,j] for j in range(parsed.shape[1]))
    reconstructed=np.stack(columns,axis=1)
    fit=reconstructed[part['fit_ids']]
    mean=np.sum(fit,axis=0)/len(fit);variance=np.sum((fit-mean)**2,axis=0)/len(fit)
    scale=np.sqrt(variance);scale[scale==0]=1
    if not np.allclose(mean,part['mean'],atol=2e-12,rtol=2e-14) or not np.allclose(scale,part['scale'],atol=2e-12,rtol=2e-14):
        raise ValueError('Saved standardizer not fit-only')
    return reconstructed
