from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


# Compatibility imports keep historical callers and saved object lookups valid.
from msc_project.baselines.candidate_tfidf import (
    PAIR_MANIFEST_COLUMNS,
    UnifiedTfidfPairConfig,
    UnifiedTfidfPairScorer,
    _text_values,
    validate_pair_manifest,
)

@dataclass(frozen=True)
class UnifiedPairCrossEncoderConfig:
    model_name: str = "distilbert-base-uncased"
    model_revision: str | None = None
    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 96
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    seed: int = 13
    use_amp: bool = True


class UnifiedPairDataset(Dataset):
    """Tokenised review/candidate sentence pairs from the shared manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        tokenizer,
        max_length: int,
        *,
        include_labels: bool,
    ) -> None:
        validate_pair_manifest(manifest, require_target=include_labels)
        self.encodings = tokenizer(
            _text_values(manifest, "text"),
            _text_values(manifest, "candidate_text"),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None
        if include_labels:
            self.labels = torch.tensor(
                pd.to_numeric(manifest["target"], errors="raise").astype(int).tolist(),
                dtype=torch.long,
            )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def set_unified_pair_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_unified_pair_tokenizer_and_model(config: UnifiedPairCrossEncoderConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=2,
        id2label={0: "absent", 1: "present"},
        label2id={"absent": 0, "present": 1},
    )
    return tokenizer, model


def make_unified_pair_train_loader(
    manifest: pd.DataFrame,
    tokenizer,
    config: UnifiedPairCrossEncoderConfig,
) -> DataLoader:
    dataset = UnifiedPairDataset(
        manifest,
        tokenizer,
        config.max_length,
        include_labels=True,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def make_unified_pair_score_loader(
    manifest: pd.DataFrame,
    tokenizer,
    config: UnifiedPairCrossEncoderConfig,
) -> DataLoader:
    dataset = UnifiedPairDataset(
        manifest,
        tokenizer,
        config.max_length,
        include_labels=False,
    )
    return DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)


def build_unified_pair_optimizer_and_scheduler(
    model,
    train_loader: DataLoader,
    config: UnifiedPairCrossEncoderConfig,
    *,
    epochs: int | None = None,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_epochs = config.epochs if epochs is None else epochs
    update_steps = max(1, len(train_loader) * total_epochs)
    warmup_steps = int(update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=update_steps,
    )
    return optimizer, scheduler


def train_unified_pair_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    config: UnifiedPairCrossEncoderConfig,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def score_unified_pair_manifest(
    model,
    tokenizer,
    manifest: pd.DataFrame,
    config: UnifiedPairCrossEncoderConfig,
    device: torch.device,
) -> np.ndarray:
    loader = make_unified_pair_score_loader(manifest, tokenizer, config)
    model.eval()
    all_scores: list[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("Unified pair cross-encoder must emit two logits per row.")
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(all_scores)


def build_unified_pair_eval_grid(
    review_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    negative_type: str = "eval_candidate",
) -> pd.DataFrame:
    """Create a stable row-major cross-product for all review/candidate pairs."""

    missing_reviews = sorted({"text", "row_uid"} - set(review_frame.columns))
    missing_candidates = sorted(
        {"candidate_text", "candidate_aspect", "candidate_sentiment"} - set(candidates.columns)
    )
    if missing_reviews:
        raise ValueError(f"Evaluation reviews are missing columns: {missing_reviews}")
    if missing_candidates:
        raise ValueError(f"Evaluation candidates are missing columns: {missing_candidates}")
    if review_frame.empty or candidates.empty:
        raise ValueError("Evaluation reviews and candidates must both be non-empty.")

    rows: list[dict[str, object]] = []
    for review in review_frame[["text", "row_uid"]].to_dict("records"):
        for candidate in candidates[
            ["candidate_text", "candidate_aspect", "candidate_sentiment"]
        ].to_dict("records"):
            rows.append({**review, **candidate, "negative_type": negative_type})
    return pd.DataFrame(rows, columns=PAIR_MANIFEST_COLUMNS)


def score_unified_pair_eval_grid(
    model,
    tokenizer,
    review_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    config: UnifiedPairCrossEncoderConfig,
    device: torch.device,
) -> pd.DataFrame:
    grid = build_unified_pair_eval_grid(review_frame, candidates)
    result = grid.copy()
    result["score"] = score_unified_pair_manifest(model, tokenizer, grid, config, device)
    return result


def fit_unified_pair_cross_encoder(
    train_manifest: pd.DataFrame,
    config: UnifiedPairCrossEncoderConfig,
    device: torch.device,
    *,
    tokenizer=None,
    model=None,
    epoch_callback: Callable[[int, float, object, object], None] | None = None,
) -> tuple[object, object, list[float]]:
    """Fit all configured epochs; injection points keep smoke tests offline."""

    validate_pair_manifest(train_manifest, require_target=True)
    set_unified_pair_seed(config.seed)
    if tokenizer is None or model is None:
        if tokenizer is not None or model is not None:
            raise ValueError("tokenizer and model must either both be supplied or both be omitted.")
        tokenizer, model = make_unified_pair_tokenizer_and_model(config)
    model.to(device)
    train_loader = make_unified_pair_train_loader(train_manifest, tokenizer, config)
    optimizer, scheduler = build_unified_pair_optimizer_and_scheduler(model, train_loader, config)
    history: list[float] = []
    for epoch_index in range(config.epochs):
        loss = train_unified_pair_epoch(model, train_loader, optimizer, scheduler, device, config)
        history.append(loss)
        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, loss, model, tokenizer)
    return tokenizer, model, history
