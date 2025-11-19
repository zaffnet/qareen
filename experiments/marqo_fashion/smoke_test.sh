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
if ! python scripts/prepare_marqo_dataset.py; then
    echo "ERROR: Dataset preparation failed"
    exit 1
fi

echo "Creating 10-sample test subset..."
if ! python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/marqo_fashion_3000')
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

    if ! python scripts/build_index.py \
        --dataset-name "$DATASET_PATH" \
        --models "$MODEL" \
        --alpha-values "$ALPHA" \
        --environment dev \
        --sample-size 10 \
        --batch-size 10 \
        --rebuild; then
        echo "ERROR: Smoke test failed for model $MODEL"
        exit 1
    fi
done

echo ""
echo "==================================================================="
echo "Smoke Test Complete! Now testing visualization..."
echo "==================================================================="
echo ""

MODEL_FLAGS=()
for MODEL in "${MODELS[@]}"; do
    MODEL_FLAGS+=("--models" "$MODEL")
done

if ! python scripts/visualize_marqo_comparison.py \
    --dataset-path "$DATASET_PATH" \
    "${MODEL_FLAGS[@]}" \
    --alpha-values "$ALPHA" \
    --environment dev \
    --k 5 \
    --output "data/marqo_smoke_test.md" \
    --seed 42; then
    echo "ERROR: Visualization generation failed"
    exit 1
fi

echo ""
echo "✓ Smoke test passed!"
echo "✓ Visualization: data/marqo_smoke_test.md"
echo ""

exit 0
