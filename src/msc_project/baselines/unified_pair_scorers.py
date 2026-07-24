from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


PAIR_MANIFEST_COLUMNS = (
    "text",
    "candidate_text",
    "candidate_aspect",
    "candidate_sentiment",
    "row_uid",
    "negative_type",
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GENERIC_CANDIDATE_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "aspect",
        "about",
        "for",
        "in",
        "is",
        "it",
        "keywords",
        "mentions",
        "of",
        "on",
        "or",
        "review",
        "sentiment",
        "the",
        "this",
        "to",
        "topic",
        "with",
    }
)
_SENTIMENT_CUES = {
    "positive": frozenset(
        {
            "amazing",
            "easy",
            "excellent",
            "fast",
            "friendly",
            "good",
            "great",
            "happy",
            "helpful",
            "love",
            "positive",
            "recommend",
            "satisfied",
        }
    ),
    "negative": frozenset(
        {
            "awful",
            "bad",
            "difficult",
            "disappointed",
            "issue",
            "negative",
            "poor",
            "problem",
            "rude",
            "slow",
            "terrible",
            "unhappy",
        }
    ),
    "neutral": frozenset({"average", "fine", "neutral", "neither", "okay", "ok"}),
}


def validate_pair_manifest(frame: pd.DataFrame, *, require_target: bool) -> None:
    """Validate the shared candidate-pair contract without mutating the input."""

    required = set(PAIR_MANIFEST_COLUMNS)
    if require_target:
        required.add("target")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate-pair manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate-pair manifest must not be empty.")
    for column in ("text", "candidate_text", "candidate_aspect", "candidate_sentiment", "row_uid"):
        if frame[column].isna().any():
            raise ValueError(f"Candidate-pair manifest column {column!r} contains missing values.")
    if require_target:
        numeric_targets = pd.to_numeric(frame["target"], errors="raise")
        targets = set(numeric_targets.tolist())
        if not targets.issubset({0, 1, 0.0, 1.0}):
            raise ValueError("Candidate-pair target values must be binary (0 or 1).")


def _text_values(frame: pd.DataFrame, column: str) -> list[str]:
    return [str(value) for value in frame[column].tolist()]


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(value).lower()))


def _token_sequence(value: object) -> list[str]:
    return _TOKEN_PATTERN.findall(str(value).lower())


def _content_tokens(value: object) -> set[str]:
    return _tokens(value) - _GENERIC_CANDIDATE_TOKENS


def _coverage(review_tokens: set[str], cue_tokens: set[str]) -> float:
    if not cue_tokens:
        return 0.0
    return len(review_tokens & cue_tokens) / len(cue_tokens)


def _cue_density(review_tokens: list[str], cue_tokens: set[str]) -> float:
    if not review_tokens:
        return 0.0
    return sum(token in cue_tokens for token in review_tokens) / len(review_tokens)


def _rowwise_cosine(left, right) -> np.ndarray:
    left = normalize(left, norm="l2", copy=False)
    right = normalize(right, norm="l2", copy=False)
    return np.asarray(left.multiply(right).sum(axis=1)).reshape(-1)


@dataclass(frozen=True)
class UnifiedTfidfPairConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 1
    max_word_features: int | None = 80_000
    max_char_features: int | None = 120_000
    classifier_c: float = 1.0
    feature_ablation: str = "all_six"
    max_iter: int = 1_000
    seed: int = 13


class UnifiedTfidfPairScorer:
    """Label-transferable candidate scorer based only on pairwise interactions.

    Both vectorisers and the logistic-regression calibration layer are fitted on
    the supplied training manifest.  No candidate identity feature is exposed
    to the classifier, so an unseen held-out aspect can be scored from its text.
    """

    feature_names = (
        "word_review_candidate_cosine",
        "character_review_candidate_cosine",
        "aspect_cue_token_coverage",
        "candidate_token_coverage",
        "sentiment_cue_token_coverage",
        "review_sentiment_cue_density",
    )
    _feature_ablation_indices = {
        "all_six": (0, 1, 2, 3, 4, 5),
        "char_cosine_and_cues": (1, 2, 3, 4, 5),
        "word_cosine_and_cues": (0, 2, 3, 4, 5),
    }

    def __init__(self, config: UnifiedTfidfPairConfig | None = None) -> None:
        self.config = config or UnifiedTfidfPairConfig()
        if self.config.feature_ablation not in self._feature_ablation_indices:
            raise ValueError(
                f"Unknown TF-IDF feature ablation: {self.config.feature_ablation!r}"
            )
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.classifier: Pipeline | None = None

    def fit(self, train_manifest: pd.DataFrame) -> "UnifiedTfidfPairScorer":
        validate_pair_manifest(train_manifest, require_target=True)
        targets = pd.to_numeric(train_manifest["target"], errors="raise").astype(int).to_numpy()
        if len(np.unique(targets)) != 2:
            raise ValueError("TF-IDF pair scorer training requires both target classes.")

        candidate_corpus = self._candidate_corpus(train_manifest)
        fit_corpus = _text_values(train_manifest, "text") + candidate_corpus
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=self.config.word_ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_word_features,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=self.config.char_ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_char_features,
            sublinear_tf=True,
        )
        self.word_vectorizer.fit(fit_corpus)
        self.char_vectorizer.fit(fit_corpus)

        features = self.transform_features(train_manifest)
        self.classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.classifier_c,
                        class_weight="balanced",
                        max_iter=self.config.max_iter,
                        solver="lbfgs",
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.classifier.fit(features, targets)
        return self

    @staticmethod
    def _candidate_corpus(manifest: pd.DataFrame) -> list[str]:
        return [
            f"{candidate_text} Aspect: {aspect}. Sentiment: {sentiment}."
            for candidate_text, aspect, sentiment in zip(
                _text_values(manifest, "candidate_text"),
                _text_values(manifest, "candidate_aspect"),
                _text_values(manifest, "candidate_sentiment"),
            )
        ]

    def transform_features(self, manifest: pd.DataFrame) -> np.ndarray:
        validate_pair_manifest(manifest, require_target=False)
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError("UnifiedTfidfPairScorer must be fitted before feature transformation.")

        reviews = _text_values(manifest, "text")
        candidates = self._candidate_corpus(manifest)
        word_cosine = _rowwise_cosine(
            self.word_vectorizer.transform(reviews),
            self.word_vectorizer.transform(candidates),
        )
        char_cosine = _rowwise_cosine(
            self.char_vectorizer.transform(reviews),
            self.char_vectorizer.transform(candidates),
        )

        candidate_coverages: list[float] = []
        aspect_coverages: list[float] = []
        sentiment_coverages: list[float] = []
        review_sentiment_densities: list[float] = []
        for review, candidate, aspect, sentiment in zip(
            reviews,
            _text_values(manifest, "candidate_text"),
            _text_values(manifest, "candidate_aspect"),
            _text_values(manifest, "candidate_sentiment"),
        ):
            review_token_sequence = _token_sequence(review)
            review_tokens = set(review_token_sequence)
            aspect_coverages.append(_coverage(review_tokens, _content_tokens(aspect)))
            candidate_coverages.append(_coverage(review_tokens, _content_tokens(candidate)))
            sentiment_key = sentiment.strip().lower()
            sentiment_cues = set(_SENTIMENT_CUES.get(sentiment_key, frozenset())) | _content_tokens(sentiment_key)
            sentiment_coverages.append(_coverage(review_tokens, sentiment_cues))
            review_sentiment_densities.append(
                _cue_density(review_token_sequence, sentiment_cues)
            )

        candidate_coverage = np.asarray(candidate_coverages, dtype=float)
        aspect_coverage = np.asarray(aspect_coverages, dtype=float)
        sentiment_coverage = np.asarray(sentiment_coverages, dtype=float)
        review_sentiment_density = np.asarray(review_sentiment_densities, dtype=float)
        full = np.column_stack(
            [
                word_cosine,
                char_cosine,
                aspect_coverage,
                candidate_coverage,
                sentiment_coverage,
                review_sentiment_density,
            ]
        )
        return full[:, self._feature_ablation_indices[self.config.feature_ablation]]

    def predict_proba(self, manifest: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("UnifiedTfidfPairScorer must be fitted before scoring.")
        return np.asarray(self.classifier.predict_proba(self.transform_features(manifest)), dtype=float)

    def score_manifest(self, manifest: pd.DataFrame) -> np.ndarray:
        """Return one positive-class probability per manifest row."""

        return self.predict_proba(manifest)[:, 1]


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
