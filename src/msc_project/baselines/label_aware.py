from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from msc_project.evaluation.metrics import PAIR_SEPARATOR, pair_to_aspect


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    seed: int = 13
    use_amp: bool = True


def candidate_pair_text(pair_label: str) -> str:
    aspect, sentiment = pair_label.split(PAIR_SEPARATOR, maxsplit=1)
    return f"Aspect: {aspect}. Sentiment: {sentiment}."


def candidate_aspect_text(aspect: str) -> str:
    leaf = aspect.split(":")[-1].strip()
    return f"Aspect: {aspect}. Topic keywords: {leaf}."


def build_pair_examples(
    frame: pd.DataFrame,
    candidate_labels: list[str],
    label_column: str,
    negatives_per_positive: int = 3,
    seed: int = 13,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    label_set = set(candidate_labels)

    for _, row in frame.iterrows():
        positives = [label for label in row[label_column] if label in label_set]
        if not positives:
            continue

        negatives = sorted(label_set - set(positives))
        sample_size = min(len(negatives), max(1, negatives_per_positive * len(positives)))
        sampled_negatives = rng.sample(negatives, sample_size) if negatives else []

        for label in positives:
            rows.append({"text": row["text"], "candidate_label": label, "target": 1})
        for label in sampled_negatives:
            rows.append({"text": row["text"], "candidate_label": label, "target": 0})

    return pd.DataFrame(rows)


def build_aspect_examples(
    frame: pd.DataFrame,
    candidate_aspects: list[str],
    label_column: str,
    negatives_per_positive: int = 3,
    seed: int = 13,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    aspect_set = set(candidate_aspects)

    for _, row in frame.iterrows():
        positives = [aspect for aspect in row[label_column] if aspect in aspect_set]
        if not positives:
            continue

        negatives = sorted(aspect_set - set(positives))
        sample_size = min(len(negatives), max(1, negatives_per_positive * len(positives)))
        sampled_negatives = rng.sample(negatives, sample_size) if negatives else []

        for aspect in positives:
            rows.append({"text": row["text"], "candidate_label": aspect, "target": 1})
        for aspect in sampled_negatives:
            rows.append({"text": row["text"], "candidate_label": aspect, "target": 0})

    return pd.DataFrame(rows)


class PairCandidateDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        tokenizer,
        max_length: int,
        include_labels: bool = True,
    ) -> None:
        self.encodings = tokenizer(
            examples["text"].tolist(),
            [candidate_pair_text(label) for label in examples["candidate_label"]],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None
        if include_labels:
            self.labels = torch.tensor(examples["target"].astype(int).tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class AspectCandidateDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        tokenizer,
        max_length: int,
        include_labels: bool = True,
    ) -> None:
        self.encodings = tokenizer(
            examples["text"].tolist(),
            [candidate_aspect_text(label) for label in examples["candidate_label"]],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None
        if include_labels:
            self.labels = torch.tensor(examples["target"].astype(int).tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def make_tokenizer_and_cross_encoder(config: CrossEncoderConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    return tokenizer, model


def make_train_loader(examples: pd.DataFrame, tokenizer, config: CrossEncoderConfig) -> DataLoader:
    dataset = PairCandidateDataset(examples, tokenizer, config.max_length, include_labels=True)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def make_score_loader(examples: pd.DataFrame, tokenizer, config: CrossEncoderConfig) -> DataLoader:
    dataset = PairCandidateDataset(examples, tokenizer, config.max_length, include_labels=False)
    return DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)


def make_aspect_train_loader(examples: pd.DataFrame, tokenizer, config: CrossEncoderConfig) -> DataLoader:
    dataset = AspectCandidateDataset(examples, tokenizer, config.max_length, include_labels=True)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def make_aspect_score_loader(examples: pd.DataFrame, tokenizer, config: CrossEncoderConfig) -> DataLoader:
    dataset = AspectCandidateDataset(examples, tokenizer, config.max_length, include_labels=False)
    return DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)


def build_optimizer_and_scheduler(model, train_loader: DataLoader, config: CrossEncoderConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=update_steps,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, device: torch.device, config: CrossEncoderConfig) -> float:
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


def make_candidate_grid(frame: pd.DataFrame, candidate_labels: list[str]) -> pd.DataFrame:
    rows = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        for label in candidate_labels:
            rows.append({"row_index": row_index, "text": row["text"], "candidate_label": label})
    return pd.DataFrame(rows)


@torch.no_grad()
def score_candidate_grid(
    model,
    tokenizer,
    frame: pd.DataFrame,
    candidate_labels: list[str],
    config: CrossEncoderConfig,
    device: torch.device,
) -> np.ndarray:
    grid = make_candidate_grid(frame, candidate_labels)
    loader = make_score_loader(grid, tokenizer, config)
    model.eval()
    scores = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        scores.append(probabilities)

    flat_scores = np.concatenate(scores)
    return flat_scores.reshape(len(frame), len(candidate_labels))


@torch.no_grad()
def score_aspect_grid(
    model,
    tokenizer,
    frame: pd.DataFrame,
    candidate_aspects: list[str],
    config: CrossEncoderConfig,
    device: torch.device,
) -> np.ndarray:
    grid = make_candidate_grid(frame, candidate_aspects)
    loader = make_aspect_score_loader(grid, tokenizer, config)
    model.eval()
    scores = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        scores.append(probabilities)

    flat_scores = np.concatenate(scores)
    return flat_scores.reshape(len(frame), len(candidate_aspects))


def predictions_from_scores(
    scores: np.ndarray,
    candidate_labels: list[str],
    threshold: float,
    ensure_one: bool = True,
    one_sentiment_per_aspect: bool = True,
    max_predictions_per_row: int | None = None,
) -> list[list[str]]:
    predictions: list[list[str]] = []

    for row_scores in scores:
        selected = np.flatnonzero(row_scores >= threshold).tolist()
        if ensure_one and not selected:
            selected = [int(np.argmax(row_scores))]

        if one_sentiment_per_aspect:
            best_by_aspect: dict[str, int] = {}
            for index in selected:
                aspect = pair_to_aspect(candidate_labels[index])
                current = best_by_aspect.get(aspect)
                if current is None or row_scores[index] > row_scores[current]:
                    best_by_aspect[aspect] = index
            selected = sorted(best_by_aspect.values())

        if max_predictions_per_row is not None and len(selected) > max_predictions_per_row:
            selected = sorted(selected, key=lambda index: row_scores[index], reverse=True)[:max_predictions_per_row]
            selected = sorted(selected)

        predictions.append([candidate_labels[index] for index in selected])

    return predictions


def aspect_predictions_from_scores(
    scores: np.ndarray,
    candidate_aspects: list[str],
    threshold: float,
    ensure_one: bool = True,
    max_predictions_per_row: int | None = None,
) -> list[list[str]]:
    predictions: list[list[str]] = []

    for row_scores in scores:
        selected = np.flatnonzero(row_scores >= threshold).tolist()
        if ensure_one and not selected:
            selected = [int(np.argmax(row_scores))]
        if max_predictions_per_row is not None and len(selected) > max_predictions_per_row:
            selected = sorted(selected, key=lambda index: row_scores[index], reverse=True)[:max_predictions_per_row]
            selected = sorted(selected)
        predictions.append([candidate_aspects[index] for index in selected])

    return predictions
