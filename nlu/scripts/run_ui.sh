#!/usr/bin/env bash
# Start We Care NLU UI on http://127.0.0.1:8088/ (host network, MiniLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NLU="$ROOT/nlu"
cd "$NLU"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WE_CARE_EMBED_OFFLINE="${WE_CARE_EMBED_OFFLINE:-1}"
export WE_CARE_SKIP_REPLY_QA="${WE_CARE_SKIP_REPLY_QA:-1}"
export WE_CARE_TRANSLATION="${WE_CARE_TRANSLATION:-mt}"
export WE_CARE_NLU_MODEL="${WE_CARE_NLU_MODEL:-st_multilingual_minilm}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true

# Stop any previous copy we own
pkill -f 'we-care-nlu serve' 2>/dev/null || true
sleep 1

mkdir -p "$NLU/logs"
echo "Starting We Care NLU UI → http://127.0.0.1:8088/"
exec "$ROOT/.venv/bin/we-care-nlu" serve --host 0.0.0.0 --port 8088
