#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /workspace
python3 - <<'PY'
import hashlib, pathlib, zipfile
archive=pathlib.Path('/workspace/policy_qwen_cloud.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '6f7a3b1145b0b64ae90e13c4c2d401515afa1ad28e0b662778f559c1bcfad3ca'
dest=pathlib.Path('/workspace/taxonomy_policy_qwen_20260908')
assert not dest.exists(), 'Existing output requires inspection'
dest.mkdir()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert (dest/name).resolve().is_relative_to(dest.resolve())
    z.extractall(dest)
print('BUNDLE_HASH_AND_EXTRACTION_PASS',flush=True)
PY
uv pip install --system --break-system-packages torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --system --break-system-packages transformers==4.57.6 bitsandbytes==0.49.2 accelerate==1.14.0 pandas==2.3.3 numpy==2.2.6 scikit-learn==1.8.0 peft==0.19.1 huggingface-hub==0.36.0 safetensors==0.7.0 psutil==7.0.0
cd /workspace/taxonomy_policy_qwen_20260908
python3 scripts/policy_qwen_cloud_worker.py --phase check
python3 scripts/policy_qwen_cloud_worker.py --phase prepare
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python3 scripts/policy_qwen_cloud_worker.py --phase run
