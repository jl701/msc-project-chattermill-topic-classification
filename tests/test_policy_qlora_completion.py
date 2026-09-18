import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

runner = module('run_taxonomy_policy_qlora_completion')
receiver = module('receive_policy_qlora_completion')
CONFIG = ROOT / 'configs/experiments/taxonomy_policy_qlora_completion_v1.json'

def config():
    return json.loads(CONFIG.read_text())

def test_frozen_contract():
    cfg = config()
    runner.validate_config(cfg)
    executor = runner.load_executor()
    assert runner.frozen_recipe(executor)['total_budget'] == 4096
    assert len(runner.l2_scopes()) == 12
    for field, value in [('include_official_test', True), ('test_contract_count', 1),
                         ('policy', 'review_filtered'), ('learning_rates', [1e-4])]:
        changed = dict(cfg, **{field: value})
        with pytest.raises(ValueError):
            runner.validate_config(changed)

def test_json_resume_tuple_and_conflict(tmp_path):
    executor = runner.load_executor()
    path = tmp_path / 'marker.json'
    value = {'target_modules': ('q_proj', 'v_proj')}
    runner._write_immutable_json(executor, path, value)
    runner._write_immutable_json(executor, path, value)
    with pytest.raises(FileExistsError):
        runner._write_immutable_json(executor, path, {'target_modules': ['v_proj']})

def grid(condition):
    return pd.DataFrame([
        {'row_uid': 'validation:0', 'candidate_aspect': 'a', 'text': 'review',
         'candidate_text': 'seen prompt', 'condition': condition, 'is_seen': True},
        {'row_uid': 'validation:0', 'candidate_aspect': 'b', 'text': 'review',
         'candidate_text': 'heldout ' + condition, 'condition': condition, 'is_seen': False}])

def test_canonical_identity_not_approximation():
    calls = []
    def score(method, runtime, rows):
        calls.append(len(rows))
        return np.arange(len(rows)) + .25, np.tile([.1, .2, .7], len(rows))
    scorer = runner.CanonicalNDScorer(score)
    n, ns = scorer('q', None, grid('N'))
    d, ds = scorer('q', None, grid('D'))
    assert calls == [2, 1]
    assert n[0] == d[0]
    assert np.array_equal(ns[:3], ds[:3])
    changed = grid('D')
    changed.loc[0, 'candidate_text'] = 'changed seen prompt'
    with pytest.raises(ValueError):
        scorer('q', None, changed)
    with pytest.raises(ValueError):
        runner.CanonicalNDScorer(score)('q', None, grid('D'))

def test_restore_N_cache():
    rows = []
    for aspect in ('a', 'b'):
        for sentiment, value in zip(runner.CANDIDATE_SENTIMENTS, [.1, .2, .7]):
            rows.append(dict(row_uid='validation:0', candidate_aspect=aspect,
                             candidate_sentiment=sentiment, aspect_score=.3, sentiment_score=value))
    scorer = runner.CanonicalNDScorer(lambda *args: (np.array([.6]), np.array([.2,.3,.5])))
    scorer.restore(pd.DataFrame(rows), grid('N'))
    scores, _ = scorer('q', None, grid('D'))
    assert scores.tolist() == [.3, .6]

@pytest.mark.parametrize('path', ['../x', '/x', 'checkpoints/../../x',
    'label_vault/x.json', 'scores/official_test.json', 'scores/CON.json', 'scores/a\\b'])
def test_receiver_path_rejection(path):
    with pytest.raises(ValueError):
        receiver.safe_artifact_path(path)

def test_double_backup_receipts_and_tamper(tmp_path):
    from msc_project.experiments.verified_artifact_sync import publish_artifact_unit, file_sha256
    remote = tmp_path / 'remote'
    remote.mkdir()
    (remote / 'scores').mkdir()
    source = remote / 'scores' / 'fixture.json'
    source.write_text('{"fixture":true}')
    pub = publish_artifact_unit(remote, ['scores/fixture.json'], unit_id='score-fixture',
        protocol_id=runner.STUDY_ID, contract_sha256='a'*64)
    manifest = json.loads(pub.read_text())
    inventory = {'transport_sha256': 'b'*64,
                 'units': {pub.name: {'manifest': manifest, 'sha256': file_sha256(pub)}}}
    roots = [tmp_path / 'first', tmp_path / 'second']
    def fetch(relative, target):
        shutil.copyfile(remote / relative, target)
    for _ in range(2):
        counts = receiver.receive_snapshot(inventory, transport_sha256='b'*64,
            local_root=roots[0], backup_root=roots[1], fetch_file=fetch)
        assert counts['receipts'] == 1
    (roots[1] / 'scores' / 'fixture.json').write_text('tampered')
    with pytest.raises(ValueError):
        receiver.receive_snapshot(inventory, transport_sha256='b'*64,
            local_root=roots[0], backup_root=roots[1], fetch_file=fetch)

def test_real_masked_scope_all_reviews_no_heldout(tmp_path):
    executor = runner.load_executor()
    data_dir = ROOT.parent / 'Project_Preparation/Public_Datasets/FABSA'
    if not (data_dir / 'train.csv').is_file():
        pytest.skip('requires the locally held FABSA training data, which is not distributed')
    args = SimpleNamespace(scope_id='heldout-a01', data_dir=data_dir, output_root=tmp_path)
    train, val, manifests, _, folds = runner.load_masked_scope(args, config(), executor)
    assert len(train) == 7930 and len(val) == 1057
    for frame in manifests.values():
        assert len(frame) == 2048
        assert not set(frame.candidate_aspect) & set(folds[0].heldout_aspects)
    runner.load_masked_scope(args, config(), executor)
