#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Virtual environment already active: $VIRTUAL_ENV"
elif [ -f ".venv/bin/activate" ] && [ -r ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment activation script not found or not readable at .venv/bin/activate" >&2
    echo "Please create and activate the virtual environment first." >&2
    exit 1
fi

DATASET_PATH="data/marqo_30k"
SAMPLE_SIZE=30
SEED=42
ENVIRONMENT="dev"
BATCH_SIZE=100
REBUILD=false

MODELS=(
    "openai/clip-vit-large-patch14"
    "Marqo/marqo-fashionSigLIP"
    # "google/siglip2-so400m-patch16-512"
    # "Marqo/marqo-ecommerce-embeddings-L"
)

ALPHA_VALUES=(0.0 0.250 0.500 0.750 1.0)

STEP="${1:-all}"

show_usage() {
    echo "Usage: $0 [STEP]"
    echo ""
    echo "Steps:"
    echo "  prepare    - Prepare dataset only"
    echo "  index      - Build indexes only (requires dataset)"
    echo "  visualize  - Generate visualization only (requires indexes)"
    echo "  all        - Run all steps (default)"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all steps"
    echo "  $0 prepare      # Only prepare dataset"
    echo "  $0 index        # Only build indexes (if dataset exists)"
    echo "  $0 visualize    # Only generate visualization (if indexes exist)"
}

if [[ "$STEP" == "help" ]] || [[ "$STEP" == "-h" ]] || [[ "$STEP" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ ! "$STEP" =~ ^(all|prepare|index|visualize)$ ]]; then
    echo "ERROR: Invalid step '$STEP'"
    echo ""
    show_usage
    exit 1
fi

echo "==================================================================="
echo "Marqo Fashion Dataset: Indexing and Visualization Pipeline"
echo "==================================================================="
echo ""
echo "Configuration:"
echo "  Dataset Path: $DATASET_PATH"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Random Seed: $SEED"
echo "  Environment: $ENVIRONMENT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Rebuild: $REBUILD"
echo "  Models: ${MODELS[*]}"
echo "  Alpha Values: ${ALPHA_VALUES[*]}"
echo "  Step: $STEP"
echo ""

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "prepare" ]]; then
    echo "Step 1: Preparing Marqo Fashion Dataset"
    echo "-------------------------------------------------------------------"
    if [ -d "$DATASET_PATH" ]; then
        echo "Dataset already exists at $DATASET_PATH, skipping preparation..."
    else
        export QAREEN_ENVIRONMENT="$ENVIRONMENT"
        export QAREEN_EMBEDDING_MODELS='["google/siglip-base-patch16-224"]'
        export QAREEN_ALPHA_VALUES='[0.5]'
        export QAREEN_DATA_DIR="data"
        export QAREEN_CHROMA_DB_DIR="chroma_db"
        export QAREEN_DATASET_PATH=""
        export QAREEN_DEV_SAMPLE_SIZE="$SAMPLE_SIZE"
        export QAREEN_BATCH_SIZE="$BATCH_SIZE"
        export QAREEN_REBUILD_COLLECTIONS="$REBUILD"
        export QAREEN_K_NEIGHBORS="5"
        export QAREEN_RANDOM_SEED="$SEED"
        export QAREEN_DATASET_PREP_SAMPLE_SIZE="$SAMPLE_SIZE"
        export QAREEN_PREPARED_DATASET_DIR="$DATASET_PATH"
        export QAREEN_VIZ_OUTPUT_FILE="data/marqo_comparison.md"
        if ! python3 scripts/prepare_marqo_dataset.py; then
            echo "ERROR: Dataset preparation failed"
            exit 1
        fi
    fi
    echo ""
fi

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "index" ]]; then
    echo "Step 2: Building Indexes for All Models and Alpha Values"
    echo "-------------------------------------------------------------------"

    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Dataset not found at $DATASET_PATH"
        echo "Run with 'prepare' or 'all' step first"
        exit 1
    fi

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "Building indexes for model: $MODEL"
        echo "-------------------------------------------------------------------"

        # Set environment variables for Settings
        export QAREEN_ENVIRONMENT="$ENVIRONMENT"
        export QAREEN_DATA_DIR="data"
        export QAREEN_CHROMA_DB_DIR="chroma_db"
        export QAREEN_DATASET_PATH=""
        export QAREEN_DEV_SAMPLE_SIZE="$SAMPLE_SIZE"
        export QAREEN_BATCH_SIZE="$BATCH_SIZE"
        export QAREEN_REBUILD_COLLECTIONS="$REBUILD"
        export QAREEN_K_NEIGHBORS="5"
        export QAREEN_RANDOM_SEED="$SEED"
        export QAREEN_DATASET_PREP_SAMPLE_SIZE="$SAMPLE_SIZE"
        export QAREEN_PREPARED_DATASET_DIR="$DATASET_PATH"
        export QAREEN_VIZ_OUTPUT_FILE="data/marqo_comparison.md"
        export QAREEN_EMBEDDING_MODELS='["'$MODEL'"]'
        export QAREEN_ALPHA_VALUES='['$(IFS=,; echo "${ALPHA_VALUES[*]}")']'
        # Build indexes using settings; models and alpha values are read from Settings
        if ! python3 scripts/build_index.py --dataset-name "$DATASET_PATH"; then
            echo "ERROR: Index building failed for model $MODEL"
            exit 1
        fi
    done
    echo ""
fi

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "visualize" ]]; then
    echo "Step 3: Generating Comparison Visualization"
    echo "-------------------------------------------------------------------"

    ALL_MODELS_JSON='['
    for i in "${!MODELS[@]}"; do
        if [ $i -gt 0 ]; then
            ALL_MODELS_JSON+=","
        fi
        ALL_MODELS_JSON+='"'"${MODELS[$i]}"'"'
    done
    ALL_MODELS_JSON+=']'

    ALL_ALPHAS_JSON='['
    for i in "${!ALPHA_VALUES[@]}"; do
        if [ $i -gt 0 ]; then
            ALL_ALPHAS_JSON+=","
        fi
        ALL_ALPHAS_JSON+="${ALPHA_VALUES[$i]}"
    done
    ALL_ALPHAS_JSON+=']'

    export QAREEN_ENVIRONMENT="$ENVIRONMENT"
    export QAREEN_DATA_DIR="data"
    export QAREEN_CHROMA_DB_DIR="chroma_db"
    export QAREEN_DATASET_PATH=""
    export QAREEN_DEV_SAMPLE_SIZE="$SAMPLE_SIZE"
    export QAREEN_BATCH_SIZE="$BATCH_SIZE"
    export QAREEN_REBUILD_COLLECTIONS="$REBUILD"
    export QAREEN_K_NEIGHBORS="5"
    export QAREEN_RANDOM_SEED="$SEED"
    export QAREEN_DATASET_PREP_SAMPLE_SIZE="$SAMPLE_SIZE"
    export QAREEN_PREPARED_DATASET_DIR="$DATASET_PATH"
    export QAREEN_EMBEDDING_MODELS="$ALL_MODELS_JSON"
    export QAREEN_ALPHA_VALUES="$ALL_ALPHAS_JSON"
    export QAREEN_VIZ_OUTPUT_FILE="data/marqo_comparison.md"

    if ! python scripts/visualize_marqo_comparison.py --dataset-path "$DATASET_PATH"; then
        echo "ERROR: Visualization generation failed"
        exit 1
    fi
    echo ""
fi

echo "==================================================================="
echo "Pipeline Complete!"
echo "==================================================================="
echo ""
echo "Results:"
echo "  - Dataset: $DATASET_PATH"
echo "  - Total Indexes: $((${#MODELS[@]} * ${#ALPHA_VALUES[@]}))"
echo "  - Visualization: data/marqo_comparison.md"
echo ""
echo "View results with: open data/marqo_comparison.md"
echo ""

exit 0
