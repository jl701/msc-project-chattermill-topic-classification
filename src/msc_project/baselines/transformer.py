from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    grad_accumulation_steps: int = 1
    seed: int = 13
    use_amp: bool = True


class FabsaTextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        binarizer: MultiLabelBinarizer,
        max_length: int,
        label_column: str = "pair_labels",
    ) -> None:
        self.encodings = tokenizer(
            frame["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            binarizer.transform(frame[label_column]),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_tokenizer_and_model(config: TransformerConfig, labels: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
    )
    return tokenizer, model


def make_dataloaders(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    tokenizer,
    labels: list[str],
    config: TransformerConfig,
    label_column: str = "pair_labels",
) -> tuple[DataLoader, DataLoader, MultiLabelBinarizer]:
    binarizer = MultiLabelBinarizer(classes=labels)
    binarizer.fit([labels])

    train_dataset = FabsaTextDataset(train_df, tokenizer, binarizer, config.max_length, label_column)
    eval_dataset = FabsaTextDataset(eval_df, tokenizer, binarizer, config.max_length, label_column)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, eval_loader, binarizer


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    config: TransformerConfig,
    pos_weight: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**batch)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(outputs.logits, labels) / config.grad_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += float(loss.detach().cpu()) * config.grad_accumulation_steps

        if step % config.grad_accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict_scores(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_true = []

    for batch in loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        scores = torch.sigmoid(outputs.logits).cpu().numpy()
        all_scores.append(scores)
        all_true.append(labels.numpy())

    return np.vstack(all_scores), np.vstack(all_true)


def build_optimizer_and_scheduler(model, train_loader: DataLoader, config: TransformerConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps = (len(train_loader) // config.grad_accumulation_steps) * config.epochs
    update_steps = max(1, update_steps)
    warmup_steps = int(update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=update_steps,
    )
    return optimizer, scheduler
