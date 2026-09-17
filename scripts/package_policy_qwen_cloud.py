"""Build a tiny allowlisted validation inference bundle, never a repo archive."""
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from msc_project.experiments.taxonomy_training_policy_sensitivity import sha256_file, write_json


def main():
    source = ROOT/'outputs/experimental/taxonomy_training_policy_sensitivity_v1'
    output = source/'cloud_handoff_20260908'
    output.mkdir(exist_ok=True)
    assert json.loads((source/'operational_recovery_20260908/pre_recovery_audit.json').read_text())['status'] == 'pass'
    files = {str(p.relative_to(ROOT)).replace('\\','/'):p for p in (ROOT/'src/msc_project').rglob('*.py')}
    for name in ['run_taxonomy_policy_qwen_sensitivity.py','campaign_taxonomy_training_policy_sensitivity.py','policy_qwen_cloud_worker.py']:
        files['scripts/'+name] = ROOT/'scripts'/name
    files['scientific_config.json'] = ROOT/'configs/experiments/taxonomy_training_policy_sensitivity_v1.json'
    for name in ['pending.jsonl','demonstrations.json','receipt.json']:
        files['prepared/'+name] = source/'qwen_prepared'/name
    transport = {'version':1,'cloud_inference_authorised':True,'include_official_test':False,
        'pending_prompts':48974,'original_science_config_unchanged':True,
        'original_infer_function_unchanged':True,'maximum_cloud_hours':6,
        'preparation_sha256':sha256_file(source/'qwen_prepared/receipt.json'),
        'files':{name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for name,p in files.items()}}
    write_json(output/'transport.json', transport)
    archive = output/'policy_qwen_cloud.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(files.items()):
            z.write(p,name)
        z.write(output/'transport.json','transport.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)+1
    write_json(output/'package_receipt.json',{'archive':archive.name,'bytes':archive.stat().st_size,
        'sha256':sha256_file(archive),'file_count':len(files)+1,'test_contract_count':0})
    print(f'PACKAGE_READY {archive} {archive.stat().st_size} bytes',flush=True)


if __name__ == '__main__':
    main()
