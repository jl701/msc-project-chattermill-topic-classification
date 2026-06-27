from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


SENTIMENTS = ("negative", "neutral", "positive")
SENTIMENT_TO_ID = {sentiment: index for index, sentiment in enumerate(SENTIMENTS)}
ID_TO_SENTIMENT = {index: sentiment for sentiment, index in SENTIMENT_TO_ID.items()}
SELECTION_METRICS = ("accuracy", "macro_f1", "micro_f1")
CLASS_WEIGHT_MODES = ("none", "balanced", "sqrt")


@dataclass(frozen=True)
class TransformerAspectSentimentConfig:
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
    class_weight: str = "balanced"
    selection_metric: str = "macro_f1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aspect_sentiment_prompt(aspect: str) -> str:
    leaf = aspect.split(":")[-1].strip()
    return f"Aspect: {aspect}. Aspect keywords: {leaf}."


def build_aspect_sentiment_examples(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        for aspect, sentiment in row["supervision_labels"]:
            rows.append(
                {
                    "text": str(row["text"]),
                    "aspect": str(aspect),
                    "sentiment": str(sentiment),
                    "target": int(SENTIMENT_TO_ID[str(sentiment)]),
                }
            )
    return pd.DataFrame(rows)


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        tokenizer,
        max_length: int,
        include_labels: bool = True,
    ) -> None:
        self.encodings = tokenizer(
            examples["text"].tolist(),
            [aspect_sentiment_prompt(aspect) for aspect in examples["aspect"]],
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


@dataclass
class TransformerAspectSentimentModel:
    tokenizer: object
    model: object
    config: TransformerAspectSentimentConfig
    device: torch.device

    def predict_pairs(self, texts: list[str], aspects: list[str]) -> list[str]:
        if len(texts) != len(aspects):
            raise ValueError("texts and aspects must have the same length.")
        if not texts:
            return []

        examples = pd.DataFrame({"text": texts, "aspect": aspects})
        dataset = AspectSentimentDataset(examples, self.tokenizer, self.config.max_length, include_labels=False)
        loader = DataLoader(dataset, batch_size=self.config.eval_batch_size, shuffle=False)
        predictions: list[str] = []
        self.model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch).logits
                predicted_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                predictions.extend(ID_TO_SENTIMENT[int(index)] for index in predicted_ids)

        return predictions


def class_weights(examples: pd.DataFrame, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"Unknown class weight mode: {mode}")
    if mode == "none":
        return None

    targets = examples["target"].astype(int).to_numpy()
    counts = np.bincount(targets, minlength=len(SENTIMENTS))
    safe_counts = np.maximum(counts, 1)
    weights = len(targets) / (len(SENTIMENTS) * safe_counts)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_optimizer_and_scheduler(model, train_loader: DataLoader, config: TransformerAspectSentimentConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=update_steps,
    )
    return optimizer, scheduler


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    config: TransformerAspectSentimentConfig,
    loss_weight: torch.Tensor | None,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=loss_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_sentiment_model(
    model,
    tokenizer,
    examples: pd.DataFrame,
    config: TransformerAspectSentimentConfig,
    device: torch.device,
) -> dict[str, float]:
    if len(examples) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "examples": 0}

    dataset = AspectSentimentDataset(examples, tokenizer, config.max_length, include_labels=True)
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    model.eval()
    gold: list[int] = []
    pred: list[int] = []

    for batch in loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        pred.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        gold.extend(labels.tolist())

    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=list(range(len(SENTIMENTS))), zero_division=0)),
        "micro_f1": float(f1_score(gold, pred, average="micro", labels=list(range(len(SENTIMENTS))), zero_division=0)),
        "examples": int(len(gold)),
    }


def selected_keys(primary_metric: str) -> tuple[str, ...]:
    if primary_metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown sentiment selection metric: {primary_metric}")
    return tuple([primary_metric] + [metric for metric in SELECTION_METRICS if metric != primary_metric])


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_transformer_aspect_sentiment_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    config: TransformerAspectSentimentConfig,
    device: torch.device,
) -> tuple[TransformerAspectSentimentModel, dict[str, object]]:
    if config.class_weight not in CLASS_WEIGHT_MODES:
        raise ValueError(f"Unknown class weight mode: {config.class_weight}")
    selected_keys(config.selection_metric)
    set_seed(config.seed)

    train_examples = build_aspect_sentiment_examples(train_df)
    validation_examples = build_aspect_sentiment_examples(validation_df)
    if len(train_examples) == 0:
        raise ValueError("Cannot train DistilBERT aspect-conditioned sentiment without training examples.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(SENTIMENTS),
        id2label={index: sentiment for index, sentiment in enumerate(SENTIMENTS)},
        label2id=SENTIMENT_TO_ID,
    )
    model.to(device)

    train_dataset = AspectSentimentDataset(train_examples, tokenizer, config.max_length, include_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)
    loss_weight = class_weights(train_examples, config.class_weight, device)

    history: list[dict[str, object]] = []
    best_epoch: dict[str, object] | None = None
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config, loss_weight)
        metrics = evaluate_sentiment_model(model, tokenizer, validation_examples, config, device)
        row = {"epoch": int(epoch), "train_loss": float(train_loss), **metrics}
        history.append(row)
        print(json.dumps({"sentiment_epoch": row}, indent=2), flush=True)

        keys = selected_keys(config.selection_metric)
        if best_epoch is None or tuple(row[key] for key in keys) > tuple(best_epoch[key] for key in keys):
            best_epoch = row
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    summary: dict[str, object] = {
        "model": "distilbert_aspect_conditioned_sentiment",
        "config": config.__dict__,
        "sentiments": list(SENTIMENTS),
        "train_examples": int(len(train_examples)),
        "validation_examples": int(len(validation_examples)),
        "train_sentiment_counts": {
            sentiment: int((train_examples["sentiment"] == sentiment).sum())
            for sentiment in SENTIMENTS
        },
        "validation_sentiment_counts": {
            sentiment: int((validation_examples["sentiment"] == sentiment).sum())
            for sentiment in SENTIMENTS
        },
        "best_validation": best_epoch or {},
        "history": history,
    }
    write_json(summary, output_dir / "sentiment_summary.json")
    return TransformerAspectSentimentModel(tokenizer, model, config, device), summary
