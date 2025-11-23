#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: Virtual environment not found at .venv"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

source .venv/bin/activate

echo "==================================================================="
echo "SMOKE TEST: 4 models × 1 alpha × 10 samples"
echo "==================================================================="
echo ""

echo "Preparing dataset..."
export QAREEN_ENVIRONMENT="dev"
export QAREEN_EMBEDDING_MODELS='["google/siglip-base-patch16-224"]'
export QAREEN_ALPHA_VALUES='[0.5]'
export QAREEN_DATA_DIR="data"
export QAREEN_CHROMA_DB_DIR="chroma_db"
export QAREEN_DATASET_PATH=""
export QAREEN_DEV_SAMPLE_SIZE="10"
export QAREEN_BATCH_SIZE="10"
export QAREEN_REBUILD_COLLECTIONS="false"
export QAREEN_K_NEIGHBORS="5"
export QAREEN_RANDOM_SEED="42"
export QAREEN_DATASET_PREP_SAMPLE_SIZE="3000"
export QAREEN_PREPARED_DATASET_DIR="data/marqo_prepared"
export QAREEN_VIZ_OUTPUT_FILE="data/marqo_smoke_test.md"
if ! python scripts/prepare_marqo_dataset.py; then
    echo "ERROR: Dataset preparation failed"
    exit 1
fi

echo "Creating 10-sample test subset..."
if ! python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/marqo_prepared')
small = dataset.select(range(10))
small.save_to_disk('data/marqo_test')
print(f'✓ Created data/marqo_test with {len(small)} samples')
"; then
    echo "ERROR: Test subset creation failed"
    exit 1
fi
echo ""

DATASET_PATH="data/marqo_test"
ALPHA=0.5

MODELS=(
    "openai/clip-vit-large-patch14"
    "Marqo/marqo-fashionSigLIP"
    "google/siglip2-so400m-patch16-512"
    "Marqo/marqo-ecommerce-embeddings-L"
)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    echo ""
    echo "[$((i+1))/4] Testing model: $MODEL"
    echo "-------------------------------------------------------------------"

    export QAREEN_ENVIRONMENT="dev"
    export QAREEN_DATA_DIR="data"
    export QAREEN_CHROMA_DB_DIR="chroma_db"
    export QAREEN_DATASET_PATH=""
    export QAREEN_DEV_SAMPLE_SIZE="10"
    export QAREEN_BATCH_SIZE="10"
    export QAREEN_REBUILD_COLLECTIONS="true"
    export QAREEN_K_NEIGHBORS="5"
    export QAREEN_RANDOM_SEED="42"
    export QAREEN_DATASET_PREP_SAMPLE_SIZE="10"
    export QAREEN_PREPARED_DATASET_DIR="data/prepared"
    export QAREEN_VIZ_OUTPUT_FILE="data/marqo_smoke_test.md"
    export QAREEN_EMBEDDING_MODELS='["'$MODEL'"]'
    export QAREEN_ALPHA_VALUES='['$ALPHA']'

    if ! python scripts/build_index.py --dataset-name "$DATASET_PATH"; then
        echo "ERROR: Smoke test failed for model $MODEL"
        exit 1
    fi
done

echo ""
echo "==================================================================="
echo "Smoke Test Complete! Now testing visualization..."
echo "==================================================================="
echo ""

ALL_MODELS_JSON='['
for i in "${!MODELS[@]}"; do
    if [ $i -gt 0 ]; then
        ALL_MODELS_JSON+=","
    fi
    ALL_MODELS_JSON+='"'"${MODELS[$i]}"'"'
done
ALL_MODELS_JSON+=']'

export QAREEN_ENVIRONMENT="dev"
export QAREEN_DATA_DIR="data"
export QAREEN_CHROMA_DB_DIR="chroma_db"
export QAREEN_DATASET_PATH=""
export QAREEN_DEV_SAMPLE_SIZE="10"
export QAREEN_BATCH_SIZE="10"
export QAREEN_REBUILD_COLLECTIONS="true"
export QAREEN_K_NEIGHBORS="5"
export QAREEN_RANDOM_SEED="42"
export QAREEN_DATASET_PREP_SAMPLE_SIZE="10"
export QAREEN_PREPARED_DATASET_DIR="data/prepared"
export QAREEN_VIZ_OUTPUT_FILE="data/marqo_smoke_test.md"
export QAREEN_EMBEDDING_MODELS="$ALL_MODELS_JSON"
export QAREEN_ALPHA_VALUES='['$ALPHA']'

if ! python scripts/visualize_marqo_comparison.py --dataset-path "$DATASET_PATH"; then
    echo "ERROR: Visualization generation failed"
    exit 1
fi

echo ""
echo "✓ Smoke test passed!"
echo "✓ Visualization: data/marqo_smoke_test.md"
echo ""

exit 0
