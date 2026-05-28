#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
MODEL_PATH="$MODEL_DIR/LFM2-700M-Q4_0.gguf"
MODEL_URL="https://huggingface.co/LiquidAI/LFM2-700M-GGUF/resolve/main/LFM2-700M-Q4_0.gguf"
EXPECTED_SIZE_MB=350
TOKENIZER_URL="https://huggingface.co/LiquidAI/LFM2-700M/resolve/main/tokenizer.json"
TOKENIZER_PATH="$MODEL_DIR/LFM2-700M-Q4_0.json"

check_dependencies() {
    if ! command -v curl &>/dev/null; then
        echo "Error: curl is required but not installed."
        exit 1
    fi
}

skip_if_exists() {
    if [ -f "$MODEL_PATH" ]; then
        local actual_size
        actual_size=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo "0")
        local expected_bytes=$((EXPECTED_SIZE_MB * 1024 * 1024))
        if [ "$actual_size" -ge "$((expected_bytes / 2))" ]; then
            echo "Model already exists at $MODEL_PATH ($((actual_size / 1024 / 1024)) MB). Skipping download."
            exit 0
        fi
        echo "Model file exists but appears incomplete ($((actual_size / 1024 / 1024)) MB). Re-downloading..."
        rm -f "$MODEL_PATH"
    fi
}

download_model() {
    echo "Downloading LFM2-700M-Q4_0.gguf from HuggingFace..."
    mkdir -p "$MODEL_DIR"
    curl -fL --retry 3 --retry-delay 5 \
        -o "$MODEL_PATH" \
        "$MODEL_URL"
    echo "Download complete."
}

verify_size() {
    local actual_size
    actual_size=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo "0")
    local actual_mb=$((actual_size / 1024 / 1024))
    local expected_bytes=$((EXPECTED_SIZE_MB * 1024 * 1024))
    local min_bytes=$((expected_bytes / 2))

    if [ "$actual_size" -lt "$min_bytes" ]; then
        echo "Error: Downloaded file is too small ($actual_mb MB). Expected at least $EXPECTED_SIZE_MB MB."
        rm -f "$MODEL_PATH"
        exit 1
    fi
    echo "Verified: $actual_mb MB downloaded successfully."
}

download_tokenizer() {
    if [ -f "$TOKENIZER_PATH" ]; then
        echo "Tokenizer already exists at $TOKENIZER_PATH. Skipping."
        return
    fi
    echo "Downloading tokenizer.json from HuggingFace..."
    curl -fL --retry 3 --retry-delay 5 \
        -o "$TOKENIZER_PATH" \
        "$TOKENIZER_URL"
    echo "Tokenizer download complete."
}

main() {
    check_dependencies
    # Always download tokenizer, even if model exists (skip_if_exists may exit early)
    download_tokenizer
    skip_if_exists
    download_model
    verify_size
    echo "Model ready at $MODEL_PATH"
}

main
