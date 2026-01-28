#!/bin/bash
# Resume benchmark on all existing models to fill in new examples

set -e

source .venv/bin/activate

MODELS=(
    # OpenAI
    "openai/gpt-5.2:off"
    "openai/gpt-5.2:medium"
    "openai/gpt-5.2:high"
    "openai/gpt-5.2:xhigh"
    "openai/gpt-5.2-pro:xhigh"

    # Google
    "google/gemini-3-pro-preview:high"
    "google/gemini-3-pro-preview:low"
    "google/gemini-3-flash-preview:high"
    "google/gemini-3-flash-preview:low"

    # Anthropic
    "anthropic/claude-opus-4.5"
    "anthropic/claude-opus-4.5:16000"

    # Others
    "z-ai/glm-4.6v"
    "x-ai/grok-4.1-fast"
    "moonshotai/kimi-k2.5"

    # Moondream (via API, not OpenRouter)
    "moondream"
    "moondream:reasoning"
)

echo "Starting resume on ${#MODELS[@]} models..."
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Running: $model"
    python harness.py --models "$model" --resume --samples 3 || echo "Warning: $model failed"
done

echo ""
echo "========================================"
echo "All done!"
