"""Read-only artifact validation, with a generated operational audit receipt."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import psutil
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    sha256_file, verify_receipt, write_json,
)


def main():
    base = ROOT / 'outputs/experimental/taxonomy_training_policy_sensitivity_v1'
    ext = ROOT / 'outputs/experimental/taxonomy_training_policy_extension_v1'
    audit = base / 'operational_recovery_20260908'
    audit.mkdir(exist_ok=True)
    states = {}
    for key, path in [('cpu', base/'campaign/cpu/state.json'),
                      ('gpu', base/'campaign/gpu/state.json'),
                      ('finish', base/'campaign/finish/state.json'),
                      ('extension', ext/'campaign/state.json')]:
        state = json.loads(path.read_text())
        assert state['failure_count'] == 0 and state['test_contract_count'] == 0, key
        for field in ('worker_pid', 'pid', 'child_pid'):
            pid = state.get(field)
            if pid and psutil.pid_exists(pid):
                raise RuntimeError(f'Process {pid} still exists: {key}')
        for name, expected in state.get('code_and_config_hashes', {}).items():
            assert sha256_file(ROOT / name) == expected, name
        states[key] = state
    assert states['cpu']['completed'] == 60
    assert states['gpu']['completed'] == 16
    assert states['extension']['completed'] == 24
    entries = []
    sets = [(base/'tfidf', 60), (base/'distilbert', 16), (ext/'dcwt', 24),
            (ext/'invariance', 48)]
    for directory, expected in sets:
        pattern = '*/*/*/receipt.json' if directory.name == 'invariance' else '*/*/receipt.json'
        receipts = sorted(directory.glob(pattern))
        assert len(receipts) == expected, (directory, len(receipts), expected)
        for path in receipts:
            source = path.parent
            observed = verify_receipt(source)
            study = base if source.is_relative_to(base) else ext
            backup = ROOT.parent/'cloud_backups'/study.name/source.relative_to(study)
            verify_receipt(backup, observed['contract'])
            assert sha256_file(path) == sha256_file(backup/'receipt.json'), path
            entries.append({'path':str(source.relative_to(ROOT)), 'files':len(observed['files']),
                            'receipt_sha256':sha256_file(path), 'backup_verified':True})
        print(f'VERIFIED {directory.name}: {expected}', flush=True)
    prepared = verify_receipt(base/'qwen_prepared')
    verify_receipt(ROOT.parent/'cloud_backups'/base.name/'qwen_prepared', prepared['contract'])
    assert not (base/'qwen_inference').exists(), 'Qwen may have started; inspect before offload'
    partial = base/'distilbert/review_filtered/l2-a09'
    assert partial.exists() and not (partial/'receipt.json').exists()
    for root in (base/'campaign/gpu', base/'campaign/finish', ext/'campaign'):
        assert not (root/'FAILED.json').exists(), root
    partial_files = {str(p.relative_to(partial)):{'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                     for p in partial.rglob('*') if p.is_file()}
    write_json(audit/'pre_recovery_audit.json', {'at':time.time(), 'status':'pass',
        'completed_receipts':len(entries), 'entries':entries, 'old_states':states,
        'partial_job_files':partial_files, 'test_contract_count':0,
        'qwen_preparation_receipt_sha256':sha256_file(base/'qwen_prepared/receipt.json')})
    print('RECOVERY_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
