#!/bin/bash

set -euo pipefail

ENV_NAME="rag_hw_env_test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yml"

create_or_update_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Updating conda environment '$ENV_NAME' from environment.yml..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        echo "Creating conda environment '$ENV_NAME' from environment.yml..."
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
    fi
}

check_env() {
    conda run -n "$ENV_NAME" python - <<'PY'
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
    "torch": "2.5.1",
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

show_next_steps() {
    cat <<EOF
=== Setup complete ===
Activate the environment with:
  conda activate $ENV_NAME

This setup installs a CUDA-capable PyTorch build for local GPU use.
If your machine has a compatible NVIDIA driver, PyTorch can use CUDA automatically.

Run one public test:
  python run.py unittest generation

Run all public tests:
  python run.py unittest all

Generate the final answers artifact locally:
  python -m solution_template
EOF
}

case "${1:-all}" in
    env)
        create_or_update_env
        ;;
    check)
        check_env
        ;;
    all|"")
        create_or_update_env
        check_env
        show_next_steps
        ;;
    *)
        echo "Usage: bash \"setup 2.sh\" [env|check|all]"
        echo "  env   - create or update the conda environment"
        echo "  check - verify that the environment imports work"
        echo "  all   - full setup (default)"
        exit 1
        ;;
esac
