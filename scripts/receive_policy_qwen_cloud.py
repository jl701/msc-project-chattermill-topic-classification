"""Receive only frozen Qwen score artifacts. Never read evaluation labels.

This transport runner does not start/restart experiments or change Pod billing.
READY_TO_STOP is published only after exact identity, hash and local-backup checks.
"""
import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, sha256_file, canonical_sha256, write_json, receipt, verify_receipt,
)
from run_taxonomy_policy_qwen_sensitivity import validate_probabilities, load_chunks
from campaign_taxonomy_training_policy_sensitivity import backup_job

REMOTE_ROOT = '/workspace/taxonomy_policy_qwen_20260908'
CHUNK = re.compile(r'chunk-[0-9]{5}\.json')
PROVENANCE = {'COMPLETE.json', 'FAILED.json', 'environment.json', 'model_manifest.json',
              'inference.log', 'telemetry.jsonl', 'state.json'}
INVENTORY = r'''
import hashlib,json,pathlib,time
root=pathlib.Path('/workspace/taxonomy_policy_qwen_20260908')
scores=root/'result/qwen_inference'
runtime=root/'runtime'
def info(p):
    return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
alive=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmd=p.read_bytes().replace(b'\0',b' ')
        if b'python' in cmd and b'policy_qwen_cloud_worker.py' in cmd:
            alive.append(int(p.parent.name))
    except (OSError,ProcessLookupError):
        pass
complete=read(runtime/'COMPLETE.json')
terminal=bool(complete and not alive and not (runtime/'worker.lock').exists())
names={p.name:info(p) for p in sorted(scores.glob('chunk-*.json'))}
if terminal:
    for name in ['state.json','receipt.json']:
        names[name]=info(scores/name)
provenance={}
if terminal or (runtime/'FAILED.json').exists():
    for name in ['COMPLETE.json','FAILED.json','environment.json','model_manifest.json','inference.log','telemetry.jsonl','state.json']:
        if (runtime/name).exists(): provenance[name]=info(runtime/name)
print(json.dumps({'at':time.time(),'files':names,'provenance':provenance,
 'state':read(scores/'state.json'),'runtime_state':read(runtime/'state.json'),
 'failure':read(runtime/'FAILED.json') or read(scores/'FAILED.json'),
 'complete':complete,'terminal':terminal,'alive_pids':alive,
 'transport_sha256':info(root/'transport.json')['sha256'] if (root/'transport.json').exists() else None}))
'''


def check_chunk(path, expected, preparation_hash, existing_keys):
    payload = json.loads(path.read_text(encoding='utf-8'))
    digest = payload.pop('payload_sha256')
    if canonical_sha256(payload) != digest or payload['preparation_sha256'] != preparation_hash:
        raise ValueError('Chunk hash/preparation conflict')
    if payload.get('test_contract_count') != 0 or payload['mode'] not in ('aspect', 'sentiment'):
        raise ValueError('Chunk boundary violation')
    probs = validate_probabilities(payload['mode'], payload['probabilities'])
    keys = payload['keys']
    if len(keys) != len(probs) or len(keys) != len(set(keys)):
        raise ValueError('Chunk row length/duplicate conflict')
    for key in keys:
        if key in existing_keys or key not in expected:
            raise ValueError('Duplicate or unexpected prompt key')
        if expected[key] != (payload['mode'], payload['demonstrations']):
            raise ValueError('Prompt mode/demonstrations mismatch')
    return set(keys)


def immutable_copy(source, destination, entry):
    if source.stat().st_size != entry['bytes'] or sha256_file(source) != entry['sha256']:
        raise ValueError('Downloaded artifact differs from remote inventory')
    if destination.exists():
        if destination.stat().st_size != entry['bytes'] or sha256_file(destination) != entry['sha256']:
            raise ValueError('Existing local artifact conflict')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.receiving')
    if temporary.exists():
        raise ValueError('Unreviewed partial local backup')
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != entry['sha256']:
        raise ValueError('Local copy hash mismatch')
    os.replace(temporary, destination)


def extract_verified_names(archive, destination, expected_names):
    with tarfile.open(archive,'r:gz') as stream:
        members = stream.getmembers()
        if len(members) != len(expected_names) or {m.name for m in members} != set(expected_names):
            raise ValueError('Transferred archive scope mismatch')
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name or member.size > 50*1024**2:
                raise ValueError('Unsafe transferred archive member')
            target = destination/member.name
            with stream.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src,dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument(
        '--identity-file',
        type=Path,
        default=os.environ.get('RUNPOD_SSH_IDENTITY_FILE'),
        help='SSH private-key path. Alternatively set RUNPOD_SSH_IDENTITY_FILE.',
    )
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    ipaddress.ip_address(args.host)
    if not 1 <= args.port <= 65535 or not re.fullmatch('[a-z0-9]+', args.pod_id):
        raise ValueError('Invalid transport destination')
    if args.identity_file is None:
        raise ValueError('Supply --identity-file or RUNPOD_SSH_IDENTITY_FILE')
    identity_file = args.identity_file.expanduser().resolve()
    if not identity_file.is_file():
        raise FileNotFoundError(f'SSH identity file not found: {identity_file}')
    base = ROOT/'outputs/experimental'/STUDY_ID
    handoff = base/'cloud_handoff_20260908'
    destination = handoff/'received'
    backup = ROOT.parent/'cloud_backups'/STUDY_ID
    destination.mkdir(exist_ok=True)
    lock = destination/'receiver.lock'
    if (destination/'FAILED.json').exists() or (destination/'READY_TO_STOP.json').exists():
        raise RuntimeError('Existing terminal receiver state requires review')
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    expected = {}
    for line in (base/'qwen_prepared/pending.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['key'] in expected:
            raise ValueError('Preparation duplicate key')
        expected[row['key']] = (row['mode'], row['demonstrations'])
    if len(expected) != 48974:
        raise ValueError('Unexpected pending scope')
    transport = json.loads((handoff/'transport.json').read_text())
    preparation_hash = sha256_file(base/'qwen_prepared/receipt.json')
    if preparation_hash != transport['preparation_sha256']:
        raise ValueError('Preparation changed')
    common = ['-i', str(identity_file), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    known, seen = {}, set()
    for path in sorted((destination/'qwen_inference').glob('chunk-*.json')):
        if not CHUNK.fullmatch(path.name):
            raise ValueError('Invalid local chunk filename')
        seen.update(check_chunk(path, expected, preparation_hash, seen))
        known[path.name] = {'bytes':path.stat().st_size,'sha256':sha256_file(path)}
    started, connection_errors = time.time(), 0
    state = {'pod_id':args.pod_id,'pid':os.getpid(),'started_unix':started,
             'test_contract_count':0,'failure_count':0,'receiver_sha256':sha256_file(Path(__file__))}
    try:
        while True:
            if time.time()-started > 6*3600:
                raise RuntimeError('Receiver six-hour deadline reached; inspect and stop Pod')
            try:
                observed = subprocess.run(['ssh',*common,'-p',str(args.port),'root@'+args.host,'python3 -'],
                    input=INVENTORY, text=True, encoding='utf-8', capture_output=True, timeout=60,
                    creationflags=flags, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                connection_errors += 1
                write_json(destination/'state.json', {**state,'status':'connection_retry',
                    'connection_errors':connection_errors,'updated_unix':time.time(),
                    'completed_prompts':len(seen),'error_type':type(error).__name__})
                if connection_errors >= 5 or args.once:
                    raise
                time.sleep(30)
                continue
            connection_errors = 0
            inventory = json.loads(observed.stdout)
            write_json(destination/'last_remote_inventory.json', inventory)
            if inventory['transport_sha256'] != sha256_file(handoff/'transport.json'):
                raise ValueError('Remote transport identity conflict')
            score_files = inventory['files']
            if not set(known).issubset(score_files):
                raise ValueError('Previously observed remote chunks disappeared')
            for name, entry in score_files.items():
                if not (CHUNK.fullmatch(name) or inventory['terminal'] and name in ('state.json','receipt.json')):
                    raise ValueError('Unexpected remote score path')
                if name in known and known[name] != entry:
                    raise ValueError('Remote sealed chunk changed')
            groups = [('qwen_inference', REMOTE_ROOT+'/result/qwen_inference',
                       {n:e for n,e in score_files.items() if n not in known}),
                      ('provenance', REMOTE_ROOT+'/runtime',inventory['provenance'])]
            for group, remote, files in groups:
                if group == 'provenance' and not set(files).issubset(PROVENANCE):
                    raise ValueError('Unexpected provenance path')
                names = list(files)
                for offset in range(0, len(names), 60):
                    batch = names[offset:offset+60]
                    stage = destination/'staging'/str(time.time_ns())/group
                    stage.mkdir(parents=True)
                    # One authenticated SSH stream per batch, not one SCP connection per file.
                    # Names are strictly allowlisted above and contain no shell metacharacters.
                    archive = stage/'transfer.tar.gz'
                    with archive.open('xb') as stream:
                        subprocess.run(['ssh',*common,'-p',str(args.port),'root@'+args.host,
                            'tar -czf - -C '+remote+' -- '+' '.join(batch)],stdout=stream,
                            stderr=subprocess.PIPE,timeout=120,creationflags=flags,check=True)
                    extract_verified_names(archive,stage,batch)
                    for name in batch:
                        path = stage/name
                        if CHUNK.fullmatch(name):
                            keys = check_chunk(path, expected, preparation_hash, seen)
                        immutable_copy(path, destination/group/name, files[name])
                        immutable_copy(path, backup/'cloud_received'/group/name, files[name])
                        if CHUNK.fullmatch(name):
                            seen.update(keys)
                            known[name] = files[name]
            state.update(status='receiving',updated_unix=time.time(),completed_prompts=len(seen),
                         planned_prompts=len(expected),chunks=len(known),
                         remote_state=inventory['state'],remote_runtime=inventory['runtime_state'])
            write_json(destination/'state.json',state)
            print(f'RECEIVED {len(seen)}/48974 chunks={len(known)}',flush=True)
            if inventory['failure']:
                raise RuntimeError('Cloud worker failure: '+repr(inventory['failure']))
            if inventory['terminal']:
                complete = inventory['complete']
                if complete['failure_count'] != 0 or complete['test_contract_count'] != 0:
                    raise ValueError('Invalid cloud completion')
                if complete['status'] != 'READY_FOR_VERIFIED_LOCAL_BACKUP' or complete['completed_prompts'] != 48974:
                    raise ValueError('Incomplete cloud boundary')
                scores = destination/'qwen_inference'
                config = json.loads((ROOT/'configs/experiments'/f'{STUDY_ID}.json').read_text())
                contract = {'study_id':STUDY_ID,'config_sha256':canonical_sha256(config),
                    'preparation_sha256':preparation_hash,
                    'runner_sha256':sha256_file(ROOT/'scripts/run_taxonomy_policy_qwen_sensitivity.py')}
                proof = verify_receipt(scores,contract)
                if set(proof['files']) | {'receipt.json'} != set(score_files):
                    raise ValueError('Receipt/inventory scope mismatch')
                if set(seen) != set(expected) or set(load_chunks(scores,preparation_hash)) != set(expected):
                    raise ValueError('Incomplete key coverage')
                if sha256_file(scores/'receipt.json') != complete['receipt_sha256']:
                    raise ValueError('Completion receipt mismatch')
                if complete['transport_sha256'] != sha256_file(handoff/'transport.json'):
                    raise ValueError('Completion transport mismatch')
                for root in [base/'qwen_inference',backup/'qwen_inference']:
                    if root.exists():
                        verify_receipt(root,contract)
                        if sha256_file(root/'receipt.json') != complete['receipt_sha256']:
                            raise ValueError('Existing final inference receipt differs')
                    backup_job(scores,root)
                provenance = destination/'provenance'
                shutil.copy2(handoff/'transport.json',provenance/'transport.json')
                receipt(provenance,{'pod_id':args.pod_id,'role':'cloud_inference_operational_provenance',
                                    'score_receipt_sha256':complete['receipt_sha256']})
                backup_job(provenance,backup/'cloud_provenance')
                ready = {**state,'status':'READY_TO_STOP','remote_processes_exited':True,
                    'remote_file_count':len(score_files),'score_receipt_sha256':complete['receipt_sha256'],
                    'provenance_receipt_sha256':sha256_file(provenance/'receipt.json'),
                    'backup_verified':True,'completed_unix':time.time()}
                write_json(destination/'READY_TO_STOP.json',ready)
                write_json(destination/'state.json',ready)
                print('LOCAL_BACKUP_VERIFIED_READY_TO_STOP '+args.pod_id,flush=True)
                return
            if args.once:
                return
            time.sleep(30)
    except BaseException as error:
        state.update(status='stopped_failure',failure_count=1,error=repr(error),updated_unix=time.time())
        write_json(destination/'FAILED.json',state)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
