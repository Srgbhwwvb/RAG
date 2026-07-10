from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np


@dataclass
class PathConfig:
    """Project-local paths used by the automated homework."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    collection_name: str = "Embeddings"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.data_dir = self.root_dir / "data"
        self.db_dir = self.root_dir / "db"


@dataclass
class ChunkingConfig:
    chunk_length: int = 500
    chunk_overlap: int = 50


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievalConfig:
    threshold: float = 0.3
    top_n: int = 3


@dataclass
class GenerationConfig:
    model_name: str | None = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 128
    fallback_answer: str = "Can't anwer with provided info"
    base_system_prompt: str = (
        "Answer the user only with the provided context. "
        "If the context is insufficient, reply exactly with "
        "\"Can't anwer with provided info\"."
    )


@dataclass
class ReferenceConfig:
    paths: PathConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    seed: int = 42


def build_reference_config(root_dir: Path | None = None) -> ReferenceConfig:
    """Create the default homework configuration."""

    paths = PathConfig(root_dir=root_dir or Path(__file__).resolve().parent)
    return ReferenceConfig(
        paths=paths,
        chunking=ChunkingConfig(),
        embedding=EmbeddingConfig(),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(),
    )


def ensure_directories(paths: PathConfig) -> None:
    """Create the directories used by the pipeline."""

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.db_dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int, *, include_torch: bool = False) -> None:
    """Seed all RNGs used by the homework pipeline."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    if not include_torch:
        return

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
