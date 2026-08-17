"""Genuine two-stage DistilBERT runtime with separately auditable task heads."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ASPECT_LABELS = {"N": 0, "Y": 1}
SENTIMENT_LABELS = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class TwoStageDistilBertConfig:
    model_id: str = "distilbert-base-uncased"
    model_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 96
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    use_amp: bool = True
    seed: int = 13

    @property
    def contract_sha256(self) -> str:
        payload = {
            "schema": "taxonomy_two_stage_distilbert_v1",
            **asdict(self),
            "aspect_labels": ASPECT_LABELS,
            "sentiment_labels": SENTIMENT_LABELS,
            "structure": "separate task heads initialized from the same pinned base",
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    task: str,
    labels: dict[str, int],
    require_answer: bool,
) -> None:
    required = {"row_uid", "text", "candidate_aspect", "candidate_text"}
    if require_answer:
        required |= {"task", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"DistilBERT {task} manifest is invalid; missing={missing}.")
    if require_answer:
        if set(frame["task"].astype(str)) != {task}:
            raise ValueError(f"DistilBERT manifest does not contain only {task} rows.")
        if set(frame["answer"].astype(str)) - set(labels):
            raise ValueError(f"DistilBERT {task} manifest has invalid answers.")
    if frame.duplicated(
        [
            "row_uid",
            "candidate_aspect",
            *( ["candidate_sentiment"] if task == "sentiment" and "candidate_sentiment" in frame else [] ),
        ]
    ).any():
        raise ValueError(f"DistilBERT {task} manifest contains duplicate identities.")


class _TextPairDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        *,
        label_map: dict[str, int] | None,
    ) -> None:
        import torch

        self.encodings = tokenizer(
            frame["text"].astype(str).tolist(),
            frame["candidate_text"].astype(str).tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None
        if label_map is not None:
            self.labels = torch.tensor(
                [label_map[str(value)] for value in frame["answer"]],
                dtype=torch.long,
            )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_classifier(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    label_map: dict[str, int],
    config: TwoStageDistilBertConfig,
    device: Any,
) -> list[dict[str, float | int]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    _set_seed(config.seed)
    dataset = _TextPairDataset(
        frame, tokenizer, config.max_length, label_map=label_map
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps = max(1, len(loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps * config.warmup_ratio),
        num_training_steps=steps,
    )
    use_amp = bool(config.use_amp and getattr(device, "type", None) == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss.detach()).item():
                raise RuntimeError("Non-finite DistilBERT two-stage training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, len(loader)),
                "batches": len(loader),
            }
        )
    return history


def _score_classifier(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: TwoStageDistilBertConfig,
    device: Any,
    *,
    width: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    dataset = _TextPairDataset(frame, tokenizer, config.max_length, label_map=None)
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.to(device)
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            if logits.ndim != 2 or logits.shape[1] != width:
                raise ValueError(f"DistilBERT task head must emit {width} logits.")
            rows.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    values = np.concatenate(rows, axis=0)
    if values.shape != (len(frame), width) or not np.isfinite(values).all():
        raise ValueError("DistilBERT two-stage probabilities are invalid.")
    return values


class DistilBertTrueTwoStageRuntime:
    method_id = "distilbert_review_candidate_cross_encoder"

    def __init__(
        self,
        config: TwoStageDistilBertConfig | None = None,
        *,
        device: Any,
        local_files_only: bool = False,
        tokenizer: Any | None = None,
        aspect_model: Any | None = None,
        sentiment_model: Any | None = None,
    ) -> None:
        self.config = config or TwoStageDistilBertConfig()
        self.device = device
        self.local_files_only = local_files_only
        self.tokenizer = tokenizer
        self.aspect_model = aspect_model
        self.sentiment_model = sentiment_model
        self.history: dict[str, list[dict[str, float | int]]] = {}

    def _load(self) -> None:
        if (
            self.tokenizer is not None
            and self.aspect_model is not None
            and self.sentiment_model is not None
        ):
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            local_files_only=self.local_files_only,
        )
        self.aspect_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            num_labels=2,
            id2label={0: "N", 1: "Y"},
            label2id=ASPECT_LABELS,
            local_files_only=self.local_files_only,
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            num_labels=3,
            id2label={0: "A", 1: "B", 2: "C"},
            label2id=SENTIMENT_LABELS,
            local_files_only=self.local_files_only,
        )

    def fit(
        self,
        aspect_manifest: pd.DataFrame,
        sentiment_manifest: pd.DataFrame,
    ) -> "DistilBertTrueTwoStageRuntime":
        _validate_manifest(
            aspect_manifest,
            task="aspect_presence",
            labels=ASPECT_LABELS,
            require_answer=True,
        )
        _validate_manifest(
            sentiment_manifest,
            task="sentiment",
            labels=SENTIMENT_LABELS,
            require_answer=True,
        )
        self._load()
        self.history = {
            "aspect_presence": _train_classifier(
                self.aspect_model,
                self.tokenizer,
                aspect_manifest,
                ASPECT_LABELS,
                self.config,
                self.device,
            ),
            "sentiment": _train_classifier(
                self.sentiment_model,
                self.tokenizer,
                sentiment_manifest,
                SENTIMENT_LABELS,
                self.config,
                self.device,
            ),
        }
        return self

    def score_aspects(self, frame: pd.DataFrame) -> np.ndarray:
        _validate_manifest(
            frame,
            task="aspect_presence",
            labels=ASPECT_LABELS,
            require_answer=False,
        )
        self._load()
        return _score_classifier(
            self.aspect_model,
            self.tokenizer,
            frame,
            self.config,
            self.device,
            width=2,
        )[:, 1]

    def score_sentiments(self, frame: pd.DataFrame) -> np.ndarray:
        _validate_manifest(
            frame,
            task="sentiment",
            labels=SENTIMENT_LABELS,
            require_answer=False,
        )
        self._load()
        return _score_classifier(
            self.sentiment_model,
            self.tokenizer,
            frame,
            self.config,
            self.device,
            width=3,
        )

    def save_pretrained(self, output_dir: Path) -> None:
        if self.tokenizer is None or self.aspect_model is None or self.sentiment_model is None:
            raise RuntimeError("Cannot save an unloaded DistilBERT two-stage runtime.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir / "tokenizer")
        self.aspect_model.save_pretrained(output_dir / "aspect_presence")
        self.sentiment_model.save_pretrained(output_dir / "sentiment")
        (output_dir / "contract.json").write_text(
            json.dumps(
                {
                    "schema_version": "taxonomy_two_stage_distilbert_checkpoint_v1",
                    "contract_sha256": self.config.contract_sha256,
                    "config": asdict(self.config),
                    "history": self.history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
