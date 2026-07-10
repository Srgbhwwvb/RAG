#!/bin/bash

set -euo pipefail
set -o xtrace

# Keep these pins aligned with environment.yml.
PIP_VERSION="25.0"
TORCH_VERSION="2.5.1"
NUMPY_VERSION="1.26.4"
SCIPY_VERSION="1.15.2"
SCIKIT_LEARN_VERSION="1.5.2"
REQUESTS_VERSION="2.32.3"
PYDANTIC_VERSION="2.10.6"
PYTEST_VERSION="8.3.5"
CHROMADB_VERSION="0.5.23"
LANGCHAIN_VERSION="0.3.19"
LANGCHAIN_COMMUNITY_VERSION="0.3.18"
LANGCHAIN_TEXT_SPLITTERS_VERSION="0.3.6"
PYMUPDF_VERSION="1.25.3"
SENTENCE_TRANSFORMERS_VERSION="3.4.1"
TRANSFORMERS_VERSION="4.46.3"

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        ;

    python3 -m pip install -qq --upgrade --ignore-installed  \
        "pip==${PIP_VERSION}" \
        ;

    python3 -m pip install -qq --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
        ;

    python3 -m pip install -qq \
        "numpy==${NUMPY_VERSION}" \
        "scipy==${SCIPY_VERSION}" \
        "scikit-learn==${SCIKIT_LEARN_VERSION}" \
        "requests==${REQUESTS_VERSION}" \
        "pydantic==${PYDANTIC_VERSION}" \
        "pytest==${PYTEST_VERSION}" \
        "chromadb==${CHROMADB_VERSION}" \
        "langchain==${LANGCHAIN_VERSION}" \
        "langchain-community==${LANGCHAIN_COMMUNITY_VERSION}" \
        "langchain-text-splitters==${LANGCHAIN_TEXT_SPLITTERS_VERSION}" \
        "pymupdf==${PYMUPDF_VERSION}" \
        "sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        ;
}

setup_checker() {
    python3 --version
    python3 -m pytest --version
    python3 - <<'PY'
from importlib import metadata

expected_versions = {
    "chromadb": "0.5.23",
    "langchain": "0.3.19",
    "langchain-community": "0.3.18",
    "langchain-text-splitters": "0.3.6",
    "numpy": "1.26.4",
    "pydantic": "2.10.6",
    "PyMuPDF": "1.25.3",
    "pytest": "8.3.5",
    "requests": "2.32.3",
    "scikit-learn": "1.5.2",
    "scipy": "1.15.2",
    "sentence-transformers": "3.4.1",
    "torch": "2.5.1+cpu",
    "transformers": "4.46.3",
}

for package_name, expected_version in expected_versions.items():
    actual_version = metadata.version(package_name)
    if actual_version != expected_version:
        raise RuntimeError(
            f"{package_name} version mismatch: expected {expected_version}, got {actual_version}"
        )

print("Environment OK")
PY
}

"$@"