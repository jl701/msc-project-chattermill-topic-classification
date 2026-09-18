"""Operational transport only. Call the unchanged frozen inference function."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import run_taxonomy_policy_qwen_sensitivity as original
from campaign_taxonomy_training_policy_sensitivity import snapshot
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    sha256_file, canonical_sha256, write_json, verify_receipt,
)


def inputs():
    transport = json.loads((ROOT/'transport.json').read_text())
    assert transport['cloud_inference_authorised'] is True
    assert transport['include_official_test'] is False
    for name, entry in transport['files'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT.resolve()), name
        assert path.stat().st_size == entry['bytes'] and sha256_file(path) == entry['sha256'], name
    config = json.loads((ROOT/'scientific_config.json').read_text())
    assert config['include_official_test'] is False and config['test_contract_count'] == 0
    prepared = ROOT/'prepared'
    receipt = json.loads((prepared/'receipt.json').read_text())
    assert receipt['contract']['config_sha256'] == canonical_sha256(config)
    assert sha256_file(prepared/'receipt.json') == transport['preparation_sha256']
    for name in ('pending.jsonl', 'demonstrations.json'):
        assert sha256_file(prepared/name) == receipt['files'][name]['sha256']
    pending = [json.loads(line) for line in (prepared/'pending.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(pending) == len({row['key'] for row in pending}) == 48974
    assert all(set(row) == {'key','mode','demonstrations','review','candidate'} for row in pending)
    assert not (ROOT/'data').exists(), 'No datasets are allowed in cloud bundle'
    return config, prepared, transport


def validate_environment():
    pins = {'torch':'2.10.0', 'transformers':'4.57.6', 'bitsandbytes':'0.49.2',
            'accelerate':'1.14.0', 'pandas':'2.3.3', 'numpy':'2.2.6',
            'scikit-learn':'1.8.0', 'peft':'0.19.1', 'huggingface-hub':'0.36.0', 'safetensors':'0.7.0'}
    observed = {p:importlib.metadata.version(p) for p in pins}
    assert all(observed[p].split('+')[0] == v for p,v in pins.items()), observed
    import torch
    assert torch.cuda.is_available()
    assert torch.version.cuda == '12.8', torch.version.cuda
    assert '4090' in torch.cuda.get_device_name(0)
    return {'packages':observed,'python':platform.python_version(),'platform':platform.platform(),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['check','prepare','run','worker'], required=True)
    args = parser.parse_args()
    config, prepared, transport = inputs()
    runtime = ROOT/'runtime'
    runtime.mkdir(exist_ok=True)
    if args.phase == 'check':
        print('ALLOWLIST_AND_48974_KEYS_VERIFIED', flush=True)
        return
    environment = validate_environment()
    if args.phase == 'prepare':
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(config['few_shot']['model_id'],
            revision=config['few_shot']['revision'],
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))
        files = {str(p.relative_to(model_dir)):{'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                 for p in sorted(model_dir.rglob('*')) if p.is_file()}
        write_json(runtime/'model_manifest.json', {'revision':config['few_shot']['revision'],'files':files})
        write_json(runtime/'environment.json', environment)
        print('PINNED_MODEL_READY', flush=True)
        return
    if args.phase == 'worker':
        original.infer(config, ROOT/'result', prepared)
        return
    assert (runtime/'model_manifest.json').exists()
    assert not (runtime/'FAILED.json').exists() and not (runtime/'COMPLETE.json').exists()
    lock = runtime/'worker.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    child = None
    started = time.time()
    thermal = 0
    try:
        with (runtime/'inference.log').open('x') as log:
            child = subprocess.Popen([sys.executable, __file__, '--phase','worker'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                gpu = snapshot()
                if float(gpu['temperature.gpu']) >= 86:
                    raise RuntimeError('GPU temperature safety stop')
                thermal += int(any(str(v).lower() == 'active' for k,v in gpu.items() if 'slowdown' in k))
                if thermal >= 2:
                    raise RuntimeError('Repeated thermal/hardware slowdown')
                if time.time()-started > 6*3600:
                    raise RuntimeError('Six-hour cloud ceiling')
                if shutil.disk_usage(ROOT).free < 20*1024**3:
                    raise RuntimeError('Disk headroom below 20 GiB')
                state = {'status':'running','pid':os.getpid(),'child_pid':child.pid,
                    'updated_unix':time.time(),'elapsed_seconds':time.time()-started,'gpu':gpu,
                    'failure_count':0,'test_contract_count':0,'role':'validation_prompt_scoring_only'}
                write_json(runtime/'state.json',state)
                with (runtime/'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(state)+'\n')
                time.sleep(15)
        if child.returncode:
            raise RuntimeError(f'Frozen inference exit {child.returncode}')
        scores = ROOT/'result/qwen_inference'
        verify_receipt(scores)
        values = original.load_chunks(scores, transport['preparation_sha256'])
        expected = {json.loads(line)['key'] for line in (prepared/'pending.jsonl').read_text().splitlines()}
        assert set(values) == expected
        import numpy as np
        for width in (2,3):
            rows = np.asarray([v for v in values.values() if len(v) == width])
            if len(rows) and np.max(np.ptp(rows,axis=0)) <= 1e-12:
                raise RuntimeError('Unexpected constant probability outputs')
        write_json(runtime/'COMPLETE.json',{'status':'READY_FOR_VERIFIED_LOCAL_BACKUP',
            'completed_prompts':len(values),'failure_count':0,'test_contract_count':0,
            'environment':environment,'transport_sha256':sha256_file(ROOT/'transport.json'),
            'receipt_sha256':sha256_file(scores/'receipt.json'),'at':time.time()})
        print('CLOUD_INFERENCE_COMPLETE_48974',flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write_json(runtime/'FAILED.json',{'error':repr(error),'at':time.time(),'failure_count':1,
                                          'test_contract_count':0})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
