"""The lexical scorer must work without importing neural libraries."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msc_project.baselines.candidate_tfidf import UnifiedTfidfPairScorer, validate_pair_manifest


def test_pre_extraction_numeric_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/tfidf_baseline.json").read_text())
    frame = pd.DataFrame(fixture["records"])
    scorer = UnifiedTfidfPairScorer().fit(frame)
    np.testing.assert_allclose(scorer.transform_features(frame), fixture["features"], atol=1e-12)
    np.testing.assert_allclose(scorer.predict_proba(frame), fixture["probabilities"], atol=1e-7)
    altered = frame.drop(columns="target").assign(negative_type="irrelevant metadata")
    np.testing.assert_array_equal(scorer.score_manifest(frame), scorer.score_manifest(altered))


def test_missing_manifest_fields_fail_early():
    with pytest.raises(ValueError, match="missing columns"):
        validate_pair_manifest(pd.DataFrame({"text": ["synthetic"]}), require_target=True)


def test_public_demo_import_does_not_load_torch():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); "
        "from msc_project.demo import run_demo; run_demo(); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
