#!/usr/bin/env bash
# FoziScribe launcher (Linux).
set -e
cd "$(dirname "$0")"

if [ -d ".venv/bin" ]; then
    source ".venv/bin/activate"
fi

# Make cuBLAS/cuDNN (pip-installed or system) visible to ctranslate2.
PYLIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
export LD_LIBRARY_PATH="/opt/cuda/lib64:${PYLIB}/nvidia/cublas/lib:${PYLIB}/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

exec python app.py
