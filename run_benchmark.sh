#!/bin/bash
# ScrupulousnessBench - Run vision models
# Automatically skips models that already have results

cd "$(dirname "$0")"

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

MODELS=(
    "openai/gpt-5.2:xhigh"
    "openai/gpt-5.2-pro:xhigh"
)

run_model() {
    local model="$1"
    local name=$(echo "$model" | sed 's|.*/||' | sed 's/:/_/g')
    local result_file="$RESULTS_DIR/${name}_results.json"

    if [[ -f "$result_file" ]]; then
        echo "SKIP: $model (already done)"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "Running: $model"
    echo "========================================"

    .venv/bin/python harness.py --models "$model" --samples 3
}

echo "ScrupulousnessBench - ${#MODELS[@]} models"
echo ""

for model in "${MODELS[@]}"; do
    run_model "$model"
done

echo ""
echo "Done! Generating plots..."
.venv/bin/python plot_results.py
