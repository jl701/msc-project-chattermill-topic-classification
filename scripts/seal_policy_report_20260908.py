"""Seal and back up completed reports only. No model or dataset execution."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    receipt, verify_receipt, sha256_file,
)
from campaign_taxonomy_training_policy_sensitivity import backup_job

PARENT = 'taxonomy_training_policy_sensitivity_v1'
EXTENSION = 'taxonomy_training_policy_extension_v1'
REPORT = ROOT.parent / 'Thesis/Training_Policy_Study_20260908'


def main():
    for study in (PARENT, EXTENSION):
        source = ROOT / 'outputs/experimental' / study / 'analysis'
        destination = REPORT / 'full_tables' / study
        payload = verify_receipt(source)
        for relative, expected in payload['files'].items():
            target = destination / relative
            if target.exists():
                if sha256_file(target) != expected['sha256']:
                    raise RuntimeError(f'Report-copy conflict: {relative}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
        if verify_receipt(destination) != payload:
            raise RuntimeError('Report analysis identity mismatch')
    for name in ('report_asset_audit.json', 'html_render_audit.json'):
        if json.loads((REPORT / name).read_text(encoding='utf-8'))['status'] != 'pass':
            raise RuntimeError('Report quality gate failed')
    operations = ROOT / 'outputs/experimental' / PARENT / 'postprocess_20260908'
    for source, name in ((operations, 'postprocess_20260908'),
                         (REPORT, 'research_report_20260908')):
        if (source / 'receipt.json').exists():
            raise RuntimeError('Already sealed. Inspect instead of overwriting.')
        payload = receipt(source, {
            'study_id': PARENT,
            'artifact_kind': name,
            'scope': 'validation_only_report_and_execution_provenance',
            'new_test_access': False,
            'seal_script_sha256': sha256_file(Path(__file__)),
        })
        destination = ROOT.parent / 'cloud_backups' / PARENT / name
        backup_job(source, destination)
        if verify_receipt(source) != verify_receipt(destination):
            raise RuntimeError('Final backup mismatch')
        print(json.dumps({'artifact': name, 'files': len(payload['files']),
                          'receipt_sha256': sha256_file(source / 'receipt.json'),
                          'backup': str(destination), 'status': 'pass'}))


if __name__ == '__main__':
    main()
