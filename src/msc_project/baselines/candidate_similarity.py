from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class CandidateSimilarityConfig:
    name: str
    family: str
    model_id: str | None = None
    revision: str | None = None
    max_length: int = 256
    batch_size: int = 64
    review_prefix: str = ""
    candidate_prefix: str = ""
    analyzer: str | None = None
    ngram_min: int | None = None
    ngram_max: int | None = None
    min_df: int | None = None
    max_features: int | None = None
    binary: bool | None = None
    sublinear_tf: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REGISTERED_CONFIGS: dict[str, CandidateSimilarityConfig] = {
    "bow_count_1_2_train_vocab": CandidateSimilarityConfig(
        name="bow_count_1_2_train_vocab",
        family="bag_of_words",
        analyzer="word",
        ngram_min=1,
        ngram_max=2,
        min_df=1,
        max_features=60000,
        binary=False,
    ),
    "tfidf_char_3_5_train_vocab": CandidateSimilarityConfig(
        name="tfidf_char_3_5_train_vocab",
        family="strict_tfidf_reference",
        analyzer="char_wb",
        ngram_min=3,
        ngram_max=5,
        min_df=1,
        max_features=None,
        sublinear_tf=True,
    ),
    "minilm_l6_v2": CandidateSimilarityConfig(
        name="minilm_l6_v2",
        family="compact_general_sentence_embedding",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        max_length=256,
        batch_size=64,
    ),
    "e5_base_v2": CandidateSimilarityConfig(
        name="e5_base_v2",
        family="stronger_retrieval_sentence_embedding",
        model_id="intfloat/e5-base-v2",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        max_length=256,
        batch_size=64,
        review_prefix="passage: ",
        candidate_prefix="query: ",
    ),
}


def resolve_configs(names: Iterable[str] | None = None) -> list[CandidateSimilarityConfig]:
    selected = list(names or REGISTERED_CONFIGS)
    unknown = sorted(set(selected) - set(REGISTERED_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown candidate-similarity configurations: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("Candidate-similarity configuration names must be unique.")
    return [REGISTERED_CONFIGS[name] for name in selected]


class SparseCandidateScorer:
    """Train-only sparse review/candidate cosine scorer."""

    def __init__(self, config: CandidateSimilarityConfig):
        if config.family == "bag_of_words":
            self.vectorizer = CountVectorizer(
                lowercase=True,
                analyzer=str(config.analyzer),
                ngram_range=(int(config.ngram_min), int(config.ngram_max)),
                min_df=int(config.min_df),
                max_features=config.max_features,
                binary=bool(config.binary),
            )
        elif config.family == "strict_tfidf_reference":
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                analyzer=str(config.analyzer),
                ngram_range=(int(config.ngram_min), int(config.ngram_max)),
                min_df=int(config.min_df),
                max_features=config.max_features,
                sublinear_tf=bool(config.sublinear_tf),
            )
        else:
            raise ValueError(f"SparseCandidateScorer does not support family {config.family!r}.")
        self.config = config
        self.fitted = False

    def fit(self, train_texts: list[str]) -> "SparseCandidateScorer":
        if not train_texts:
            raise ValueError("Sparse candidate scorer requires at least one training review.")
        # Candidate, validation, and test text must never participate in fitting.
        self.vectorizer.fit([str(text) for text in train_texts])
        self.fitted = True
        return self

    def score(self, eval_texts: list[str], candidate_text: str) -> tuple[np.ndarray, dict[str, object]]:
        if not self.fitted:
            raise RuntimeError("SparseCandidateScorer must be fitted before scoring.")
        review_matrix = normalize(self.vectorizer.transform([str(text) for text in eval_texts]))
        candidate_matrix = normalize(self.vectorizer.transform([str(candidate_text)]))
        scores = np.asarray((review_matrix @ candidate_matrix.T).toarray(), dtype=float).reshape(-1)
        diagnostics = {
            "vocabulary_size": int(len(self.vectorizer.vocabulary_)),
            "candidate_vector_nnz": int(candidate_matrix.nnz),
            "candidate_zero_vector": bool(candidate_matrix.nnz == 0),
            "candidate_vector_norm": float(np.sqrt(candidate_matrix.multiply(candidate_matrix).sum())),
        }
        return scores, diagnostics


def masked_mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    token_sum = (last_hidden_state * mask).sum(dim=1)
    token_count = mask.sum(dim=1).clamp(min=1e-9)
    return token_sum / token_count


class FrozenTransformerSentenceEncoder:
    """Pinned frozen encoder using the model-card mean-pooling recipe."""

    def __init__(
        self,
        config: CandidateSimilarityConfig,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        if not config.model_id or not config.revision:
            raise ValueError("Frozen sentence encoders require a model ID and pinned revision.")

        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.config = config
        self.device = torch.device(resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: list[str], role: str) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if role not in {"review", "candidate"}:
            raise ValueError(f"Unknown embedding role: {role}")
        prefix = self.config.review_prefix if role == "review" else self.config.candidate_prefix
        formatted = [f"{prefix}{str(text)}" for text in texts]
        if not formatted:
            hidden_size = int(getattr(self.model.config, "hidden_size"))
            return np.empty((0, hidden_size), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(formatted), self.config.batch_size):
            batch = formatted[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded)
                pooled = masked_mean_pool(output.last_hidden_state, encoded["attention_mask"])
                pooled = functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    def score_matrix(self, review_texts: list[str], candidate_texts: list[str]) -> tuple[np.ndarray, dict[str, object]]:
        review_embeddings = self.encode(review_texts, role="review")
        candidate_embeddings = self.encode(candidate_texts, role="candidate")
        scores = np.asarray(review_embeddings @ candidate_embeddings.T, dtype=float)
        diagnostics = {
            "embedding_dimension": int(review_embeddings.shape[1]),
            "review_rows_encoded": int(review_embeddings.shape[0]),
            "candidate_rows_encoded": int(candidate_embeddings.shape[0]),
            "device": str(self.device),
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "max_length": int(self.config.max_length),
            "pooling": "attention_mask_mean",
            "normalisation": "l2",
        }
        return scores, diagnostics

    def close(self) -> None:
        import torch

        self.model.to("cpu")
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
