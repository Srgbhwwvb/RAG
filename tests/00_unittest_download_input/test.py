from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import build_pdf_bytes, load_module


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


def test_download_pdfs_writes_named_files(tmp_path, monkeypatch):
    module = load_module("data_ingestion")
    urls = [
        "https://example.com/history/part_1.pdf",
        "https://example.com/history/part_2.pdf",
    ]
    payloads = {
        urls[0]: build_pdf_bytes(["alpha beta gamma"]),
        urls[1]: build_pdf_bytes(["delta epsilon zeta"]),
    }
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _FakeResponse(payloads[url])

    monkeypatch.setattr(module.requests, "get", fake_get)

    paths = module.download_pdfs(urls, tmp_path)

    expected_paths = [tmp_path / "part_1.pdf", tmp_path / "part_2.pdf"]
    assert paths == expected_paths
    assert all(path.exists() for path in expected_paths)
    assert len(calls) == 2
