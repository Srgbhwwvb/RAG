from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import DEFAULT_QUESTIONS, load_module

def _build_db(tmp_path):
    preprocessing = load_module("preprocessing")
    vector_db_module = load_module("vector_db")
    return vector_db_module.VectorDB(
        ROOT,
        {".pdf": preprocessing.PyMuPDFLoader},
        500,
        50,
        db_path=tmp_path / "db",
    )


def _build_filled_db(tmp_path):
    db = _build_db(tmp_path)
    db.fill_db(reset_collection=True)
    return db


def test_fill_db_populates_collection(tmp_path):
    db = _build_db(tmp_path)

    added = db.fill_db(reset_collection=True)

    assert isinstance(added, int)
    assert added > 0
    assert db.collection.count() > 0


def test_query_without_rerank_returns_relevant_context(tmp_path):
    db = _build_filled_db(tmp_path)

    docs, scores = db.query(DEFAULT_QUESTIONS[0], 0.3, 3, use_rerank=False)

    assert docs is not None
    assert scores is not None
    assert len(docs) >= 1
    assert len(scores) == len(docs)


class _FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs):
        return list(range(len(pairs), 0, -1))


def test_rerank_uses_mocked_cross_encoder(tmp_path, monkeypatch):
    vector_db_module = load_module("vector_db")
    monkeypatch.setattr(vector_db_module, "CrossEncoder", _FakeCrossEncoder)

    db = _build_db(tmp_path)
    candidate_documents = [
        "Lenin's funeral, January 1924",
        "The first Russian state was Kievan Rus.",
        "Russia is currently ruled by Vladimir Putin.",
    ]
    docs, scores = db.rerank(
        DEFAULT_QUESTIONS[0],
        candidate_documents,
        1,
        "fake-cross-encoder",
    )

    assert docs is not None
    assert scores is not None
    assert len(docs) == 1
    assert len(scores) == 1
