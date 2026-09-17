"""Independent local close-out audit. Scores only, no evaluation labels."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, sha256_file, canonical_sha256, verify_receipt, write_json,
)
from receive_policy_qwen_cloud import check_chunk
from run_taxonomy_policy_qwen_sensitivity import load_chunks


def incremental():
    base = ROOT/'outputs/experimental'/STUDY_ID
    scores = base/'cloud_handoff_20260908/received/qwen_inference'
    values = load_chunks(scores,sha256_file(base/'qwen_prepared/receipt.json'))
    report = {'role':'label_free_operational_check','prompts':len(values),'modes':{}}
    for width in (2,3):
        rows = np.asarray([v for v in values.values() if len(v) == width])
        if not len(rows):
            continue
        result = {'rows':len(rows),'finite':bool(np.isfinite(rows).all()),
            'maximum_column_range':float(np.max(np.ptp(rows,axis=0))),
            'argmax_counts':np.bincount(rows.argmax(axis=1),minlength=width).tolist()}
        assert result['finite'] and result['maximum_column_range'] > 1e-12
        report['modes'][str(width)] = result
    print(json.dumps(report),flush=True)


def main():
    base = ROOT/'outputs/experimental'/STUDY_ID
    handoff = base/'cloud_handoff_20260908'
    received = handoff/'received'
    backup = ROOT.parent/'cloud_backups'/STUDY_ID
    ready = json.loads((received/'READY_TO_STOP.json').read_text())
    if (received/'FAILED.json').exists():
        raise ValueError('Unresolved current transport failure')
    assert ready['status'] == 'READY_TO_STOP' and ready['remote_processes_exited']
    assert ready['failure_count'] == ready['test_contract_count'] == 0
    assert ready['backup_verified'] and ready['completed_prompts'] == 48974
    assert ready['pod_id'] == 'y9yh6i4r9y5ph4'
    transport = json.loads((handoff/'transport.json').read_text())
    archive = json.loads((handoff/'package_receipt.json').read_text())
    assert sha256_file(handoff/archive['archive']) == archive['sha256']
    assert archive['sha256'] == '6f7a3b1145b0b64ae90e13c4c2d401515afa1ad28e0b662778f559c1bcfad3ca'
    for name,entry in transport['files'].items():
        if name == 'scientific_config.json':
            path = ROOT/'configs/experiments'/f'{STUDY_ID}.json'
        elif name.startswith('prepared/'):
            path = base/'qwen_prepared'/name.split('/',1)[1]
        else:
            path = ROOT/name
        assert path.stat().st_size == entry['bytes'] and sha256_file(path) == entry['sha256'], name
    prepared = base/'qwen_prepared'
    verify_receipt(prepared)
    preparation_hash = sha256_file(prepared/'receipt.json')
    assert preparation_hash == transport['preparation_sha256']
    config = json.loads((ROOT/'configs/experiments'/f'{STUDY_ID}.json').read_text())
    assert config['include_official_test'] is False and config['test_contract_count'] == 0
    contract = {'study_id':STUDY_ID,'config_sha256':canonical_sha256(config),
        'preparation_sha256':preparation_hash,
        'runner_sha256':sha256_file(ROOT/'scripts/run_taxonomy_policy_qwen_sensitivity.py')}
    inventory = json.loads((received/'last_remote_inventory.json').read_text())
    assert inventory['terminal'] and not inventory['alive_pids'] and not inventory['failure']
    roots = [received/'qwen_inference',base/'qwen_inference',backup/'qwen_inference']
    for root in roots:
        proof = verify_receipt(root,contract)
        assert sha256_file(root/'receipt.json') == ready['score_receipt_sha256']
        assert set(proof['files']) | {'receipt.json'} == set(inventory['files'])
        assert {p.name for p in root.iterdir() if p.is_file()} == set(inventory['files'])
        for name,entry in inventory['files'].items():
            assert sha256_file(root/name) == entry['sha256']
    expected = {}
    for line in (prepared/'pending.jsonl').read_text(encoding='utf-8').splitlines():
        value = json.loads(line)
        assert value['key'] not in expected
        expected[value['key']] = (value['mode'],value['demonstrations'])
    seen = set()
    for path in sorted(roots[0].glob('chunk-*.json')):
        seen.update(check_chunk(path,expected,preparation_hash,seen))
    assert len(seen) == 48974 and seen == set(expected)
    for root in [received/'provenance',backup/'cloud_provenance']:
        verify_receipt(root)
        assert sha256_file(root/'receipt.json') == ready['provenance_receipt_sha256']
        assert sha256_file(root/'transport.json') == sha256_file(handoff/'transport.json')
        model = json.loads((root/'model_manifest.json').read_text())
        assert model['revision'] == config['few_shot']['revision']
        assert any(name.endswith('.safetensors') for name in model['files'])
    result = {'status':'pass','pod_id':ready['pod_id'],'prompt_count':len(seen),
        'score_file_count':len(inventory['files']),'verified_score_copies':len(roots),
        'score_receipt_sha256':ready['score_receipt_sha256'],
        'provenance_receipt_sha256':ready['provenance_receipt_sha256'],
        'remote_processes_exited':True,'ready_to_stop':True,'test_contract_count':0,
        'failure_count':0,'official_test_accessed':False,'performance_metrics_computed':False,
        'archived_transport_timeout':'FAILED_scp_v1.json, not an inference failure',
        'auditor_sha256':sha256_file(Path(__file__)),'audited_unix':time.time()}
    output = handoff/'completion_audit.json'
    if output.exists():
        raise ValueError('Existing close-out audit must not be overwritten')
    write_json(output,result)
    write_json(backup/'cloud_completion_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--incremental']:
        incremental()
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('Unsupported audit arguments')
