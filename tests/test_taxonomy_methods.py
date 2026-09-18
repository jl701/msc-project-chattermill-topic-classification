from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_methods import (
    E5PairRuntime,
    METHOD_IDS,
    QwenPairRuntime,
    StrictTfidfRuntime,
    distilbert_config_from_parameters,
    load_method_registry,
    method_registry_sha256,
    qlora_training_config_from_parameters,
    resolve_method_spec,
    tfidf_config_from_parameters,
    validate_probability_scores,
)
from msc_project.llm.qwen_pair_classifier import VerbalizerTokenIds


def pair_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_uid": "r1",
                "text": "email support was good",
                "candidate_text": "email support positive",
                "candidate_aspect": "Support: Email",
                "candidate_sentiment": "positive",
                "negative_type": "gold_positive",
                "target": 1,
            },
            {
                "row_uid": "r1",
                "text": "email support was good",
                "candidate_text": "phone support positive",
                "candidate_aspect": "Support: Phone",
                "candidate_sentiment": "positive",
                "negative_type": "absent",
                "target": 0,
            },
            {
                "row_uid": "r2",
                "text": "phone support was terrible",
                "candidate_text": "phone support negative",
                "candidate_aspect": "Support: Phone",
                "candidate_sentiment": "negative",
                "negative_type": "gold_positive",
                "target": 1,
            },
            {
                "row_uid": "r2",
                "text": "phone support was terrible",
                "candidate_text": "email support negative",
                "candidate_aspect": "Support: Email",
                "candidate_sentiment": "negative",
                "negative_type": "absent",
                "target": 0,
            },
        ]
    )


def test_registry_has_exact_five_ordered_methods_and_matched_qwen_base() -> None:
    registry = load_method_registry()
    assert tuple(registry["methods"]) == METHOD_IDS
    assert len(method_registry_sha256(registry)) == 64
    frozen = resolve_method_spec("frozen_qwen_candidate_pair", registry)
    qlora = resolve_method_spec("qwen_candidate_pair_qlora", registry)
    assert frozen.model_id == qlora.model_id
    assert frozen.model_revision == qlora.model_revision
    assert frozen.requires_pair_training is False
    assert qlora.requires_pair_training is True


def test_registry_parameters_create_exact_runtime_configs() -> None:
    registry = load_method_registry()
    tfidf_spec = resolve_method_spec("strict_train_only_tfidf", registry)
    tfidf = tfidf_config_from_parameters(tfidf_spec.starting_recipe)
    assert tfidf.feature_ablation == "all_six"
    assert tfidf.classifier_c == pytest.approx(1.0)

    distil_spec = resolve_method_spec(
        "distilbert_review_candidate_cross_encoder",
        registry,
    )
    distil_parameters = {
        **distil_spec.starting_recipe,
        "selected_checkpoint_epoch": 2,
    }
    distil = distilbert_config_from_parameters(distil_parameters, distil_spec)
    assert distil.model_revision == distil_spec.model_revision
    assert distil.epochs == 2

    qlora_spec = resolve_method_spec("qwen_candidate_pair_qlora", registry)
    qlora = qlora_training_config_from_parameters(
        {**qlora_spec.starting_recipe, "selected_checkpoint_epoch": 1}
    )
    assert qlora.epochs == 1
    assert qlora.gradient_accumulation_steps == 8


def test_probability_contract_rejects_shape_range_and_nan() -> None:
    assert validate_probability_scores([0.0, 1.0], expected_rows=2).shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        validate_probability_scores([[0.2], [0.3]], expected_rows=2)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        validate_probability_scores([-0.1, 0.4], expected_rows=2)
    with pytest.raises(ValueError, match="finite"):
        validate_probability_scores([np.nan, 0.4], expected_rows=2)


def test_strict_tfidf_fit_never_learns_eval_only_candidate_token() -> None:
    train = pair_manifest()
    runtime = StrictTfidfRuntime().fit(train)
    evaluation = train.iloc[[0]].drop(columns="target").copy()
    evaluation["candidate_text"] = "neverfittedtoken positive"
    evaluation["candidate_aspect"] = "Unseen: Never fitted"
    scores = runtime.score(evaluation)
    assert scores.shape == (1,)
    assert "neverfittedtoken" not in runtime.scorer.word_vectorizer.vocabulary_


class FakeE5:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def score_matrix(self, reviews, candidates):
        self.calls.append((list(reviews), list(candidates)))
        matrix = np.arange(len(reviews) * len(candidates), dtype=float).reshape(
            len(reviews), len(candidates)
        )
        matrix = np.clip(matrix / max(1, matrix.size - 1), -1, 1)
        return matrix, {}

    def close(self):
        self.closed = True


def test_e5_runtime_deduplicates_encoding_and_restores_pair_order() -> None:
    manifest = pair_manifest()
    encoder = FakeE5()
    runtime = E5PairRuntime(encoder=encoder).fit(manifest)
    scores = runtime.score(manifest.drop(columns="target"))
    assert scores.shape == (len(manifest),)
    assert len(encoder.calls[0][0]) == 2
    assert len(encoder.calls[0][1]) == 4
    runtime.close()
    assert encoder.closed


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        return "prompt tokens"

    def __call__(self, text, add_special_tokens=False):
        if text == "Y":
            return {"input_ids": [3]}
        if text == "N":
            return {"input_ids": [4]}
        return {"input_ids": [5, 6]}


class FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        logits[:, :, 3] = 2.0
        logits[:, :, 4] = 0.0
        return SimpleNamespace(logits=logits)


def test_qwen_frozen_and_qlora_share_score_interface_but_not_fit_behaviour() -> None:
    manifest = pair_manifest()
    ids = VerbalizerTokenIds(yes=3, no=4)
    frozen = QwenPairRuntime(
        method_id="frozen_qwen_candidate_pair",
        model=FakeQwen(),
        tokenizer=FakeTokenizer(),
        verbalizer_ids=ids,
        batch_size=2,
    )
    frozen.fit(manifest)
    frozen_scores = frozen.score(manifest.drop(columns="target"))

    calls = []

    def fit_callback(frame):
        calls.append(len(frame))
        return FakeQwen(), FakeTokenizer(), ids

    qlora = QwenPairRuntime(
        method_id="qwen_candidate_pair_qlora",
        model=FakeQwen(),
        tokenizer=FakeTokenizer(),
        verbalizer_ids=ids,
        batch_size=2,
        fit_callback=fit_callback,
    )
    with pytest.raises(RuntimeError, match="fitted"):
        qlora.score(manifest.drop(columns="target"))
    qlora.fit(manifest)
    qlora_scores = qlora.score(manifest.drop(columns="target"))
    assert calls == [len(manifest)]
    np.testing.assert_allclose(frozen_scores, qlora_scores)
