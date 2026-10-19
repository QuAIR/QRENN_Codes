"""Official preparation and independent fit-only encoding audits; no model fits."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from . import data, train, evaluate

OUT=Path(os.environ.get('QRENN_EVIDENCE_ROOT','/Users/mingrui/Projects/QRENN/output'))/'prxq-a07-shoppers-rice-2026-10-18'


def test_new_campaign_inventory():
    p=train.protocol()
    assert p['datasets']==['shoppers','rice']
    tasks=train.make_tasks()
    assert len(tasks)==30 and len(evaluate.expected_coordinates())==100
    assert p['evaluation']['selected_counts']==dict(qrenn=30,kernel=120,classical=40,total=190)
    assert {t['dataset']:t['n'] for t in tasks}==dict(shoppers=7,rice=3)
    assert all(t['seed']==82000+100*t['resample']+t['init'] and t['shuffle_seed']==t['seed']+1000000 for t in tasks)
    assert train.primary_metric('shoppers')=='average_precision' and train.primary_metric('rice')=='roc_auc'


def test_fit_only_categories_unknowns_and_reconstruction():
    from . import additional_data as extra
    x=np.zeros((4,17));x[:,10:]=np.array([[0]*7,[1]*7,[2]*7,[1]*7]);x[:,0]=[1,3,5,7]
    categories=extra.fit_categories(x,np.array([0,1]))
    assert all(np.array_equal(c,[0,1]) for c in categories.values())
    encoded=extra.transform_categories(x,categories)
    assert encoded.shape==(4,24) and not encoded[2,10:].any()
    np.testing.assert_array_equal(encoded[0,10:],np.tile([1,0],7))
    np.testing.assert_array_equal(encoded[1,10:],np.tile([0,1],7))


def test_official_additional_preparation(tmp_path):
    manifest=data.prepare(OUT/'raw',tmp_path)
    assert list(manifest['datasets'])==['shoppers','rice'] and len(manifest['splits'])==10
    expected=json.loads((OUT/'dataset-preflight.json').read_text())
    for name in ['shoppers','rice']:
        spec=manifest['datasets'][name];target=expected[name]
        assert (spec['rows'],spec['positive'],spec['unique_feature_groups'])==(target['rows'],target['positive'],target['groups'])
        with np.load(tmp_path/(name+'.npz'),allow_pickle=False) as raw:
            assert raw['original_labels'].dtype.kind=='U'
            assert raw['raw_tokens'].shape==(target['rows'],target['original_features'])
            if name=='shoppers':assert json.loads(str(raw['token_dictionary_json']))==target['raw_token_dictionary']==spec['raw_token_dictionary']
        for r,part in enumerate(target['splits']):
            split=data.load_split(tmp_path,name,r)
            assert 'test' not in split and split['fit']['x'].shape[1]==part['expanded_features']
            complete=data.load_split(tmp_path,name,r,include_test=True)
            for section in ['fit','validation','test']:
                assert len(complete[section]['ids'])==part[section]
            assert all(not np.intersect1d(complete[a]['groups'],complete[b]['groups']).size for a,b in [('fit','validation'),('fit','test'),('validation','test')])
            audit=data.audit_reconstruction(tmp_path,name,r)
            assert audit['encoded_columns_checked']==part['expanded_features'] and audit['fit_only_statistics_verified']
    # Saved vocabularies are authenticated even if a changed encoding keeps the dimension.
    path=tmp_path/'shoppers-split-0.npz'
    with np.load(path) as arrays:changed={key:arrays[key] for key in arrays.files}
    changed['category_10']=changed['category_10'][::-1]
    np.savez_compressed(path,**changed)
    with pytest.raises(ValueError,match='checksum'):data.load_split(tmp_path,'shoppers',0)


@pytest.mark.parametrize('names',[['rice','rice'],['unknown']])
def test_preparation_rejects_duplicate_or_unknown_dataset_names(tmp_path,monkeypatch,names):
    original=data.protocol
    monkeypatch.setattr(data,'protocol',lambda:dict(original(),datasets=names))
    with pytest.raises(ValueError,match='dataset'):data.prepare(OUT/'raw',tmp_path)


def test_literal_counts_and_complete_inventory_are_consistent(monkeypatch):
    records=[]
    methods={'qrenn':['qrenn_zz'],'kernel':['fidelity','projected'],
             'classical':['lr_standardized','lr_angles','rbf_svm','gradient_boosting']}
    for coordinate in evaluate.expected_coordinates():records.append(dict(coordinate=list(coordinate),methods=methods[coordinate[0]]))
    evaluate.check_formal_inventory(records,190)
    for changed in (records[:-1],records+[records[0]]):
        with pytest.raises(ValueError,match='formal|declared'):evaluate.check_formal_inventory(changed,190)
    bad=[dict(r) for r in records];bad[0]=dict(bad[0],methods=[])
    with pytest.raises(ValueError,match='formal|declared'):evaluate.check_formal_inventory(bad,190)
    literal=train.protocol
    def inconsistent():
        p=literal();p['evaluation']['selected_counts']['kernel']=119;return p
    monkeypatch.setattr(train,'protocol',inconsistent)
    with pytest.raises(ValueError,match='formal|declared'):evaluate.check_formal_inventory(records,190)


def test_training_cli_uses_literal_new_datasets(capsys,monkeypatch):
    import sys
    monkeypatch.setattr(sys,'argv',['train','--help'])
    with pytest.raises(SystemExit) as result:train.main()
    assert result.value.code==0
    assert '{shoppers,rice}' in capsys.readouterr().out


def test_manual_audit_rejects_leaked_vocabulary_and_statistics():
    from . import additional_data as extra
    x=np.zeros((3,17));x[:,10:]=np.array([[0]*7,[1]*7,[2]*7])
    raw=dict(x=x,raw_tokens=np.asarray([[str(v) for v in row] for row in x]),
             feature_names=np.asarray([str(j) for j in range(17)]),
             token_dictionary_json=np.asarray(json.dumps({str(j):['0.0','1.0','2.0'] for j in range(10,17)})))
    part=dict(fit_ids=np.array([0,1]),mean=np.r_[np.zeros(10),np.tile([.5,.5],7)],
              scale=np.r_[np.ones(10),np.full(14,.5)],**{f'category_{j}':np.array([0.,1.]) for j in range(10,17)})
    assert extra.manual_reconstruction(raw,part,'shoppers').shape==(3,24)
    leaked=dict(part,category_10=np.array([0.,1.,2.]))
    with pytest.raises(ValueError,match='fit-only'):extra.manual_reconstruction(raw,leaked,'shoppers')
    with pytest.raises(ValueError,match='fit-only'):extra.manual_reconstruction(raw,dict(part,mean=part['mean']+1),'shoppers')
