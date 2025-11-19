#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

source .venv/bin/activate

DATASET_PATH="data/marqo_fashion_3000"
SAMPLE_SIZE=3000
SEED=42
ENVIRONMENT="dev"
BATCH_SIZE=100

MODELS=(
    "openai/clip-vit-large-patch14"
    "Marqo/marqo-fashionSigLIP"
    "google/siglip2-so400m-patch16-512"
    "Marqo/marqo-ecommerce-embeddings-L"
)

ALPHA_VALUES=(0.0 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.0)

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
        if ! python scripts/prepare_marqo_dataset.py; then
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

        ALPHA_FLAGS=""
        for ALPHA in "${ALPHA_VALUES[@]}"; do
            ALPHA_FLAGS="$ALPHA_FLAGS --alpha-values $ALPHA"
        done

        if ! python scripts/build_index.py \
            --dataset-name "$DATASET_PATH" \
            --models "$MODEL" \
            "$ALPHA_FLAGS" \
            --environment "$ENVIRONMENT" \
            --sample-size $SAMPLE_SIZE \
            --batch-size $BATCH_SIZE; then
            echo "ERROR: Index building failed for model $MODEL"
            exit 1
        fi
    done
    echo ""
fi

if [[ "$STEP" == "all" ]] || [[ "$STEP" == "visualize" ]]; then
    echo "Step 3: Generating Comparison Visualization"
    echo "-------------------------------------------------------------------"

    MODEL_FLAGS=""
    for MODEL in "${MODELS[@]}"; do
        MODEL_FLAGS="$MODEL_FLAGS --models $MODEL"
    done

    ALPHA_FLAGS=""
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        ALPHA_FLAGS="$ALPHA_FLAGS --alpha-values $ALPHA"
    done

    if ! python scripts/visualize_marqo_comparison.py \
        --dataset-path "$DATASET_PATH" \
        "$MODEL_FLAGS" \
        "$ALPHA_FLAGS" \
        --environment "$ENVIRONMENT" \
        --k 5 \
        --output "data/marqo_comparison.md" \
        --seed $SEED; then
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
