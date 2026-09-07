#!/usr/bin/env bash
# Plain-curl tour of the API. Assumes the server is on 127.0.0.1:8000.
set -euo pipefail
BASE="${MOBILEMOE_BASE_URL:-http://127.0.0.1:8000}"
AUTH=()
if [ -n "${MOBILEMOE_API_KEY:-}" ]; then
  AUTH=(-H "Authorization: Bearer ${MOBILEMOE_API_KEY}")
fi

echo "--- health"
curl -s "${BASE}/health"; echo

echo "--- models"
curl -s "${AUTH[@]}" "${BASE}/v1/models"; echo

echo "--- chat completion"
curl -s "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MobileMoE-S-QAT",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Why are open-source on-device language models great?"}
    ],
    "max_tokens": 128
  }'; echo

echo "--- streaming"
curl -sN "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MobileMoE-S-QAT",
    "messages": [{"role": "user", "content": "Count from one to ten in words."}],
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'; echo

echo "--- sampling with a stop string"
curl -s "${AUTH[@]}" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write one line about the sea."}],
    "temperature": 0.8, "top_p": 0.95, "seed": 7, "stop": ["\n"], "max_tokens": 48
  }'; echo

echo "--- legacy completions"
curl -s "${AUTH[@]}" "${BASE}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The three primary colours are", "max_tokens": 24}'; echo
