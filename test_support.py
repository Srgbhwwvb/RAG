from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import fitz
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT
IMPLEMENTATION_PACKAGE = os.environ.get("RAG_HW_IMPL_PACKAGE", "")

DEFAULT_QUESTIONS = (
    "What was the first Russian state?",
    "Who was the last king of Poland?",
    "Who does rule Russia currently?",
    "How many people did Russia lose because of the separate peace in 1918?",
    "What is capital of Finland?",
)

EXPECTED_UUIDS = (
    "b1d16c1a701703dd14b0b6f470e61c6c7b4c5b97568d553e064999d18859f3c2",
    "db7a34d94b9b9c8e55f41dda9d0b4f980f327008f25947436f042a51d81223af",
)

DETERMINISTIC_EMBEDDING_DIM = 32
EMBEDDING_TEST_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_TEST_TEXT = "Silly Sally sells sea shells"
EXPECTED_TEST_EMBEDDING = np.asarray(
    [
        0.0,
        0.0,
        0.30151134729385376,
        0.0,
        0.30151134729385376,
        0.0,
        0.0,
        0.6030226945877075,
        0.30151134729385376,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.30151134729385376,
        0.0,
        0.0,
        0.0,
        0.0,
        0.30151134729385376,
        0.30151134729385376,
        0.0,
        0.0,
        0.30151134729385376,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
FALLBACK_ANSWER = "Can't anwer with provided info"


class DeterministicSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.vectorizer = HashingVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            norm="l2",
            alternate_sign=False,
            stop_words="english",
            n_features=DETERMINISTIC_EMBEDDING_DIM,
        )

    def encode(
        self,
        texts,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        del show_progress_bar

        embeddings = self.vectorizer.transform(list(texts)).toarray().astype(np.float32)
        if normalize_embeddings and embeddings.size:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            embeddings = embeddings / norms

        if convert_to_numpy:
            return embeddings
        return embeddings.tolist()


def load_module(module_name: str):
    full_name = f"{IMPLEMENTATION_PACKAGE}.{module_name}" if IMPLEMENTATION_PACKAGE else module_name

    module = importlib.import_module(full_name)

    try:
        embedding_candidates = []
        if IMPLEMENTATION_PACKAGE:
            embedding_candidates.append(f"{IMPLEMENTATION_PACKAGE}.embedding")
        else:
            embedding_candidates.append("embedding")

        for emb_name in embedding_candidates:
            try:
                embedding_module = importlib.import_module(emb_name)
            except ImportError:
                continue
            embedding_module.SentenceTransformer = DeterministicSentenceTransformer
            embedding_module.transformer_cls = DeterministicSentenceTransformer
    except ImportError:
        pass
    return module


def build_pdf_bytes(lines: list[str]) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 520, 760), "\n\n".join(lines), fontsize=14)
    payload = document.tobytes()
    document.close()
    return payload


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def answer_matches(question: str, answer: str) -> bool:
    normalized = normalize_answer(answer)

    if question == DEFAULT_QUESTIONS[0]:
        return ("kievan" in normalized or "kiev" in normalized) and "rus" in normalized
    if question == DEFAULT_QUESTIONS[1]:
        return "poniatowski" in normalized
    if question == DEFAULT_QUESTIONS[2]:
        return "vladimir" in normalized and "putin" in normalized
    if question == DEFAULT_QUESTIONS[3]:
        return "34%" in normalized or "55 million" in normalized
    if question == DEFAULT_QUESTIONS[4]:
        return "can't anwer with provided info" in normalized
    raise ValueError(f"Unknown question: {question}")
