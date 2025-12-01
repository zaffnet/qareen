#!/bin/bash

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f .venv/bin/activate ] || { echo "ERROR: Virtual environment not found" >&2; exit 1; }
source .venv/bin/activate

export QAREEN_DATASET_PATH="data/marqo_prepared" QAREEN_ALPHA_VALUES="[0.0, 0.25, 0.5, 0.75, 1.0]"
MODELS=("openai/clip-vit-large-patch14" "Marqo/marqo-fashionSigLIP" "google/siglip2-so400m-patch14-224" "Marqo/marqo-ecommerce-embeddings-B")
NUM_ALPHAS=$(python -c "import json; print(len(set(json.loads('$QAREEN_ALPHA_VALUES'))))")
NUM_MODELS=${#MODELS[@]}
MODELS_JSON=$(python -c "import json, sys; print(json.dumps(sys.argv[1:]))" "${MODELS[@]}")

set_common_env() {
    export QAREEN_ENVIRONMENT="dev" QAREEN_DATA_DIR="data" QAREEN_CHROMA_DB_DIR="chroma_db"
    export QAREEN_EMBEDDING_MODELS="$MODELS_JSON" QAREEN_DEV_SAMPLE_SIZE="300" QAREEN_BATCH_SIZE="100"
    export QAREEN_K_NEIGHBORS="10" QAREEN_RANDOM_SEED="43" QAREEN_VIZ_OUTPUT_FILE="data/marqo_smoke_test.md"
    export QAREEN_PREPARED_DATASET_DIR="data/marqo_prepared" QAREEN_DATASET_PREP_SAMPLE_SIZE="300"
    export QAREEN_REBUILD_COLLECTIONS="true"
}

set_common_env

echo "==================================================================="
echo "SMOKE TEST: ${NUM_MODELS} models × ${NUM_ALPHAS} alphas"
echo "==================================================================="

set_common_env && python scripts/prepare_marqo_dataset.py || { echo "ERROR: Dataset preparation failed" >&2; exit 1; }
set_common_env && python scripts/build_index.py || { echo "ERROR: Index building failed" >&2; exit 1; }
set_common_env && python scripts/visualize_marqo_comparison.py || { echo "ERROR: Visualization failed" >&2; exit 1; }

echo "✓ Smoke test passed! Visualization: data/marqo_smoke_test.md"
