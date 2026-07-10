from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_PDF_URLS = (
    "https://publichealth.hsc.wvu.edu/media/5553/russian-history-part-i.pdf",
    "https://publichealth.hsc.wvu.edu/media/5589/history-of-russia-part-2.pdf",
)


def download_pdfs(
    pdf_urls: tuple[str, ...] | list[str],
    data_dir: Path,
    timeout: int = 60,
) -> list[Path]:
    """Download the PDFs into the local data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []

    for url in pdf_urls:
        parsed = urlparse(url)
        path = parsed.path
        filename = Path(path).name

        if not filename:
            raise ValueError(f"Cannot derive a filename from URL: {url}")

        local_path = data_dir / filename

        if local_path.exists():
            downloaded_paths.append(local_path)
            continue

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  

        with open(local_path, "wb") as f:
            f.write(response.content)

        downloaded_paths.append(local_path)

    return downloaded_paths


if __name__ == "__main__":
    download_pdfs(DEFAULT_PDF_URLS, Path("."))