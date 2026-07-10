from pathlib import Path
import sys

from langchain_core.documents import Document

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import DATA_DIR, load_module


def _build_db(tmp_path):
    preprocessing = load_module("preprocessing")
    vector_db_module = load_module("vector_db")
    return vector_db_module.VectorDB(
        DATA_DIR,
        {".pdf": preprocessing.PyMuPDFLoader},
        500,
        50,
        db_path=tmp_path / "db",
    )


def test_get_uuids_returns_deterministic_string_ids(tmp_path):
    db = _build_db(tmp_path)
    chunks = db.preprocess_data()

    uuids = db.get_uuids(chunks)

    assert len(uuids) == len(chunks)
    assert all(isinstance(uid, str) and uid for uid in uuids)
    assert len(set(uuids)) == len(uuids)


def test_get_uuids_changes_when_content_changes(tmp_path):
    db = _build_db(tmp_path)
    chunks = db.preprocess_data()
    modified = Document(
        page_content=chunks[0].page_content + " extra text",
        metadata=dict(chunks[0].metadata),
    )

    original_uuid = db.get_uuids([chunks[0]])[0]
    modified_uuid = db.get_uuids([modified])[0]

    assert original_uuid != modified_uuid
