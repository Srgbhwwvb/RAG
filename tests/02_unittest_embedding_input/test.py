from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import (
    EMBEDDING_TEST_MODEL_NAME,
    EMBEDDING_TEST_TEXT,
    EXPECTED_TEST_EMBEDDING,
    load_module,
)

class _Chunk:
    def __init__(self, text: str) -> None:
        self.page_content = text


def test_get_embeddings_returns_expected_values_for_fixed_text():
    module = load_module("embedding")
    chunks = [_Chunk(EMBEDDING_TEST_TEXT)]
    embedder = module.SentenceEmbedder(model_name=EMBEDDING_TEST_MODEL_NAME)
    embeddings = embedder.get_embeddings(chunks)

    assert embeddings.shape == (1, EXPECTED_TEST_EMBEDDING.shape[0])
    np.testing.assert_allclose(embeddings[0], EXPECTED_TEST_EMBEDDING, rtol=0.0, atol=1e-7)
