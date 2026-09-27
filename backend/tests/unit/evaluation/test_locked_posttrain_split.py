import json
import pytest
from scripts.collect_harness_batch import validate_locked_test_release,validate_case_partition
from evaluation.posttrain_scale_curriculum import TRAIN_VERSION,TEST_VERSION


def test_training_cases_cannot_enter_locked_test_or_dev():
    case={'split':'train','dataset_version':TRAIN_VERSION}
    assert validate_case_partition([case],'train')==TRAIN_VERSION
    for split in ['validation','test']:
        with pytest.raises(ValueError):validate_case_partition([case],split)


def test_test_cases_cannot_become_training_cases():
    case={'split':'test','dataset_version':TEST_VERSION}
    assert validate_case_partition([case],'test')==TEST_VERSION
    with pytest.raises(ValueError):validate_case_partition([case],'train')


@pytest.mark.parametrize('field,value',[('schema_version','wrong'),('cases_sha256','other'),('source_hashes',{}),('models',[])])
def test_locked_release_rejects_changed_inputs_or_unlisted_model(tmp_path,field,value):
    config={'cases_sha256':'cases','source_hashes':{'loop.py':'source'},'weight_hashes':{'adapter':'weights'}}
    release={'schema_version':'locked-test-release.v1',**config,'models':[{'weight_hashes':config['weight_hashes']}]}
    release[field]=value;p=tmp_path/'release.json';p.write_text(json.dumps(release))
    with pytest.raises(ValueError):validate_locked_test_release(config,p)


def test_locked_release_requires_manifest_and_accepts_exact_frozen_model(tmp_path):
    config={'cases_sha256':'cases','source_hashes':{'loop.py':'source'},'weight_hashes':{'adapter':'weights'}}
    with pytest.raises(ValueError):validate_locked_test_release(config,None)
    p=tmp_path/'release.json';p.write_text(json.dumps({'schema_version':'locked-test-release.v1',**config,'models':[{'weight_hashes':config['weight_hashes']}]}))
    assert len(validate_locked_test_release(config,p))==64
