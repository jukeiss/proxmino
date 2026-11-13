#!/usr/bin/env bash
# From your Mac: copy Proxmino to the server and run the installer there.
#   ./deploy/send-to-server.sh you@casaos.local [model or models/your-model.gguf]
# Copies the app, your .env keys (the installer moves them into the app's private data volume) and,
# if you name a .gguf file, that model. Leaves out your chats and settings on this Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
SERVER=${1:?usage: ./deploy/send-to-server.sh you@server [model]}
MODEL=${2:-phi4-mini}
rsync -a --delete --exclude .venv --exclude data --exclude __pycache__ --exclude .pytest_cache --exclude 'models/*.gguf' ./ "$SERVER:proxmino/"
if [[ "$MODEL" == *.gguf ]]; then
  ssh "$SERVER" mkdir -p proxmino/models
  rsync -a --progress "$MODEL" "$(dirname "$MODEL")/Modelfile.$(basename "$MODEL" | sed -E 's/-q4_k_m\.gguf$|\.gguf$//')" "$SERVER:proxmino/models/" 2>/dev/null || rsync -a --progress "$MODEL" "$SERVER:proxmino/models/"
  MODEL=models/$(basename "$MODEL")
fi
ssh -t "$SERVER" "cd proxmino && ./deploy/install-casaos.sh '$MODEL'"
