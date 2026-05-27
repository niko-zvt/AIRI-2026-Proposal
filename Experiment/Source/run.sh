#!/usr/bin/env bash
# Единая точка входа. Ставит venv + moftransformer, скачивает чекпойнт
# и базы QMOF/CoREMOF/hMOF (~7 GB при первом запуске), прогоняет 00..05.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SOURCE_DIR}/.." && pwd)"
RESULTS_DIR="${EXPERIMENT_DIR}/Results"
VENV_DIR="${SOURCE_DIR}/.venv"
PY_BIN="python3.10"

mkdir -p "${RESULTS_DIR}"

log() { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }

# venv: uv по возможности, иначе fallback на python -m venv
HAS_UV=0
if command -v uv >/dev/null 2>&1; then
    HAS_UV=1
elif "${PY_BIN}" -m pip install --user --quiet uv >/dev/null 2>&1; then
    USER_BIN="$("${PY_BIN}" -c 'import site; print(site.USER_BASE)')/bin"
    export PATH="${USER_BIN}:${PATH}"
    if command -v uv >/dev/null 2>&1; then
        HAS_UV=1
    fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    if [[ "${HAS_UV}" -eq 1 ]]; then
        log "Создаю venv через uv (Python 3.10) в ${VENV_DIR}"
        uv venv --python 3.10 "${VENV_DIR}"
    else
        log "uv недоступен, fallback на python -m venv"
        "${PY_BIN}" -m venv "${VENV_DIR}"
    fi
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! python -c "import torch" >/dev/null 2>&1; then
    log "PyTorch не найден - ставлю CUDA 12.4 wheel"
    if [[ "${HAS_UV}" -eq 1 ]]; then
        uv pip install --index-url https://download.pytorch.org/whl/cu124 \
            "torch>=2.5.0,<2.7.0" "torchvision"
    else
        pip install --index-url https://download.pytorch.org/whl/cu124 \
            "torch>=2.5.0,<2.7.0" "torchvision"
    fi
fi

log "Ставлю зависимости проекта"
if [[ "${HAS_UV}" -eq 1 ]]; then
    (cd "${SOURCE_DIR}" && uv pip install -r pyproject.toml)
else
    pip install --upgrade pip
    pip install -r "${SOURCE_DIR}/requirements.txt"
fi

# moftransformer на PyPI пиннит torch<2.0 / PL==1.7, но сам код работает
# с torch 2.x. Ставим --no-deps, транзитивные зависимости уже выше.
if ! python -c "import moftransformer" >/dev/null 2>&1; then
    log "Ставлю moftransformer==2.2.0 (--no-deps)"
    if [[ "${HAS_UV}" -eq 1 ]]; then
        uv pip install --no-deps "moftransformer==2.2.0"
    else
        pip install --no-deps "moftransformer==2.2.0"
    fi
fi

# Все базы кладутся внутрь .venv/.../moftransformer/database/.
PMT_CKPT="$(python -c 'import moftransformer, os; print(os.path.join(moftransformer.__root_dir__, "database", "pmtransformer.ckpt"))')"
if [[ ! -s "${PMT_CKPT}" ]]; then
    log "Скачиваю PMTransformer checkpoint (~50 MB)"
    moftransformer download pretrain_model
else
    log "PMTransformer checkpoint уже на месте"
fi

QMOF_FLAG="$(python -c 'import moftransformer, os; print(os.path.join(moftransformer.__root_dir__, "database", "qmof", "raw", "ABACUF01_FSR.grid"))')"
if [[ ! -s "${QMOF_FLAG}" ]]; then
    log "Скачиваю QMOF (~2.5 GB)"
    moftransformer download qmof -r
else
    log "QMOF уже на месте"
fi

COREMOF_FLAG="$(python -c 'import moftransformer, os; print(os.path.join(moftransformer.__root_dir__, "database", "coremof", "raw", "ABAVIJ_clean.grid"))')"
if [[ ! -s "${COREMOF_FLAG}" ]]; then
    log "Скачиваю CoREMOF (~1.6 GB)"
    moftransformer download coremof -r
else
    log "CoREMOF уже на месте"
fi

# `moftransformer download hmof` использует wget, который не справляется
# с AWS-WAF challenge на figshare и создаёт пустой tar.gz. Качаем curl'ом.
HMOF_DIR="$(python -c 'import moftransformer, os; print(os.path.join(moftransformer.__root_dir__, "database", "hmof"))')"
HMOF_FLAG="${HMOF_DIR}/downstream_release/train_raspa_100bar.json"
if [[ ! -s "${HMOF_FLAG}" ]]; then
    log "Скачиваю hMOF (~3 GB) через curl"
    mkdir -p "${HMOF_DIR}"
    HMOF_TAR="${HMOF_DIR}/hmof.tar.gz"
    rm -f "${HMOF_TAR}"
    curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 \
        -o "${HMOF_TAR}" \
        "https://ndownloader.figshare.com/files/37511755"
    log "Распаковываю hMOF"
    tar -xzf "${HMOF_TAR}" -C "${HMOF_DIR}"
    rm -f "${HMOF_TAR}"
else
    log "hMOF уже на месте"
fi

cd "${SOURCE_DIR}"
log "Шаг 00 - выбор пула (1500 QMOF + 1500 CoREMOF + 1500 hMOF)"
python 00_select_pool.py

log "Шаг 01 - извлечение эмбеддингов PMTransformer"
python 01_extract_embeddings.py

log "Шаг 02 - расчёт proxy-дескрипторов"
python 02_compute_proxy.py

log "Шаг 03 - визуализация латента (PCA / UMAP / t-SNE)"
python 03_visualize_latent.py

log "Шаг 04 - k-NN retrieval, метрики, baseline'ы"
python 04_retrieval_eval.py

#log "Шаг 05 - сборка отчёта"
#python 05_make_report.py

log "Готово. Отчёт: ${RESULTS_DIR}/report.md"
