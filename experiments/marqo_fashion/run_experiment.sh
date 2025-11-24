#!/bin/bash

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -f .venv/bin/activate ] || { echo "ERROR: Virtual environment not found" >&2; exit 1; }
source .venv/bin/activate

DATASET_PATH="data/marqo_30k"
SAMPLE_SIZE=30000
SEED=42
ENVIRONMENT="prod"
BATCH_SIZE=100
REBUILD=false
MODELS=("openai/clip-vit-large-patch14" "Marqo/marqo-fashionSigLIP")
ALPHA_VALUES=(0.0 0.250 0.500 0.750 1.0)
STEP="${1:-all}"

show_usage() {
    echo "Usage: $0 [STEP]"
    echo "Steps: prepare | index | visualize | all (default)"
}

[[ "$STEP" =~ ^(help|-h|--help)$ ]] && { show_usage; exit 0; }
[[ "$STEP" =~ ^(all|prepare|index|visualize)$ ]] || { echo "ERROR: Invalid step '$STEP'" >&2; show_usage; exit 1; }

echo "==================================================================="
echo "Marqo Fashion Dataset: Indexing and Visualization Pipeline"
echo "==================================================================="
echo "Dataset: $DATASET_PATH | Sample: $SAMPLE_SIZE | Env: $ENVIRONMENT | Batch: $BATCH_SIZE | Rebuild: $REBUILD"
echo "Models: ${MODELS[*]} | Alpha: ${ALPHA_VALUES[*]} | Step: $STEP"
echo ""

set_common_env() {
    export QAREEN_ENVIRONMENT="$ENVIRONMENT" QAREEN_DATA_DIR="data" QAREEN_CHROMA_DB_DIR="chroma_db"
    export QAREEN_DATASET_PATH="" QAREEN_DEV_SAMPLE_SIZE="$SAMPLE_SIZE" QAREEN_BATCH_SIZE="$BATCH_SIZE"
    export QAREEN_REBUILD_COLLECTIONS="$REBUILD" QAREEN_K_NEIGHBORS="5" QAREEN_RANDOM_SEED="$SEED"
    export QAREEN_DATASET_PREP_SAMPLE_SIZE="$SAMPLE_SIZE" QAREEN_PREPARED_DATASET_DIR="$DATASET_PATH"
    export QAREEN_VIZ_OUTPUT_FILE="data/marqo_comparison.md"
}

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "prepare" ]]; then
    echo "Step 1: Preparing Marqo Fashion Dataset"
    [ -d "$DATASET_PATH" ] && echo "Dataset exists, skipping..." || {
        set_common_env
        export QAREEN_EMBEDDING_MODELS='["google/siglip-base-patch16-224"]' QAREEN_ALPHA_VALUES='[0.5]'
        python3 scripts/prepare_marqo_dataset.py || { echo "ERROR: Dataset preparation failed" >&2; exit 1; }
    }
fi

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "index" ]]; then
    echo "Step 2: Building Indexes"
    [ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset not found at $DATASET_PATH" >&2; exit 1; }
    for MODEL in "${MODELS[@]}"; do
        echo "Building indexes for: $MODEL"
        set_common_env
        export QAREEN_EMBEDDING_MODELS='["'$MODEL'"]'
        export QAREEN_ALPHA_VALUES="[$(IFS=,; echo "${ALPHA_VALUES[*]}")]"
        python3 scripts/build_index.py --dataset-name "$DATASET_PATH" || { echo "ERROR: Index building failed for $MODEL" >&2; exit 1; }
    done
fi

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "visualize" ]]; then
    echo "Step 3: Generating Visualization"
    MODELS_JSON=$(python3 -c "import json, sys; print(json.dumps(sys.argv[1:]))" "${MODELS[@]}")
    ALPHAS_JSON=$(python3 -c "import json, sys; print(json.dumps([float(x) for x in sys.argv[1:]]))" "${ALPHA_VALUES[@]}")
    set_common_env
    export QAREEN_EMBEDDING_MODELS="$MODELS_JSON" QAREEN_ALPHA_VALUES="$ALPHAS_JSON"
    python scripts/visualize_marqo_comparison.py --dataset-path "$DATASET_PATH" || { echo "ERROR: Visualization failed" >&2; exit 1; }
fi

echo "==================================================================="
echo "Pipeline Complete! Indexes: $((${#MODELS[@]} * ${#ALPHA_VALUES[@]})) | Visualization: data/marqo_comparison.md"
echo "==================================================================="
