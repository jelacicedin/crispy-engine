#!/usr/bin/env bash
# Downloads the GPT-2 small weights and tokenizer files into models/gpt2/.
# Pinned to a fixed Hugging Face commit so the reference never changes underneath us.
set -euo pipefail

REVISION=607a30d783dfa663caf39e06633721c8d4cfcd7e
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/gpt2"

hf download openai-community/gpt2 \
    model.safetensors config.json generation_config.json \
    tokenizer.json tokenizer_config.json vocab.json merges.txt \
    --revision "$REVISION" \
    --local-dir "$DEST"
