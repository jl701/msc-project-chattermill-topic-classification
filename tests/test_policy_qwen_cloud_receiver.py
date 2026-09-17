"""Synthetic transport tests, with no dataset access or model execution."""
import importlib.util
import json
import sys
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('policy_receiver', ROOT/'scripts/receive_policy_qwen_cloud.py')
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)


def chunk(tmp_path, **overrides):
    value = {'mode':'aspect', 'demonstrations':'demo', 'keys':['key1'],
             'probabilities':[[0.6,0.4]], 'preparation_sha256':'prep', 'test_contract_count':0}
    value.update(overrides)
    value['payload_sha256'] = receiver.canonical_sha256(value)
    path = tmp_path/'chunk-00000.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_valid_chunk(tmp_path):
    assert receiver.check_chunk(chunk(tmp_path), {'key1':('aspect','demo')}, 'prep', set()) == {'key1'}


@pytest.mark.parametrize('change', [
    {'preparation_sha256':'wrong'}, {'test_contract_count':1}, {'mode':'other'},
    {'keys':['unknown']}, {'keys':['key1','key1'], 'probabilities':[[.6,.4],[.6,.4]]},
    {'demonstrations':'wrong'}, {'probabilities':[[.7,.7]]}, {'probabilities':[[.2,.3,.5]]},
])
def test_invalid_chunk_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        receiver.check_chunk(chunk(tmp_path,**change), {'key1':('aspect','demo')}, 'prep', set())


def test_duplicate_across_chunks_rejected(tmp_path):
    with pytest.raises(ValueError):
        receiver.check_chunk(chunk(tmp_path), {'key1':('aspect','demo')}, 'prep', {'key1'})


def test_corrupt_payload_rejected(tmp_path):
    path = chunk(tmp_path)
    value = json.loads(path.read_text())
    value['probabilities'] = [[.4,.6]]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        receiver.check_chunk(path, {'key1':('aspect','demo')}, 'prep', set())


def test_immutable_copy_and_conflict(tmp_path):
    source = tmp_path/'source.json'
    source.write_text('{}')
    entry = {'bytes':source.stat().st_size, 'sha256':receiver.sha256_file(source)}
    target = tmp_path/'backup/file.json'
    receiver.immutable_copy(source,target,entry)
    receiver.immutable_copy(source,target,entry)
    assert receiver.sha256_file(target) == entry['sha256']
    target.write_text('changed')
    with pytest.raises(ValueError):
        receiver.immutable_copy(source,target,entry)


def test_remote_inventory_compiles_without_execution():
    compile(receiver.INVENTORY,'remote_inventory','exec')


def test_batched_transfer_valid_and_scope_rejection(tmp_path):
    archive = tmp_path/'transfer.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        member = tarfile.TarInfo('chunk-00000.json')
        member.size = 2
        stream.addfile(member,io.BytesIO(b'{}'))
    with pytest.raises(ValueError):
        receiver.extract_verified_names(archive,tmp_path,['unexpected.json'])
    receiver.extract_verified_names(archive,tmp_path,['chunk-00000.json'])
    assert (tmp_path/'chunk-00000.json').read_bytes() == b'{}'


def test_batched_transfer_rejects_symlink(tmp_path):
    archive = tmp_path/'transfer.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        member = tarfile.TarInfo('chunk-00000.json')
        member.type = tarfile.SYMTYPE
        member.linkname = '/etc/passwd'
        stream.addfile(member)
    with pytest.raises(ValueError):
        receiver.extract_verified_names(archive,tmp_path,['chunk-00000.json'])
