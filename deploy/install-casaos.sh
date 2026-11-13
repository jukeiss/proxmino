#!/usr/bin/env bash
# Install or update Proxmino on a CasaOS (or any Docker) server. Run it on the server, in the Proxmino folder:
#   ./deploy/install-casaos.sh              # with phi4-mini as the local model
#   ./deploy/install-casaos.sh granite3.3:8b                         # or any model from ollama.com/library
#   ./deploy/install-casaos.sh models/proxmino-8b-q4_k_m.gguf        # or your own model file (from export_model.sh)
# It builds the app, starts it next to its own Ollama, downloads the model, moves your keys into the app's
# data volume and asks for a sign-in password. Proxmino then shows up in the CasaOS dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-phi4-mini}
DC="docker compose -f deploy/docker-compose.yml"
IP=$( (hostname -I 2>/dev/null || true) | awk '{print $1}')
HOST=$(hostname); HOST=${HOST%.local}
export SEARXNG_SECRET=${SEARXNG_SECRET:-$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d " \n")}
export PROXMINO_HOSTS=${PROXMINO_HOSTS:-$HOST.local${IP:+,$IP}}

echo "Building and starting Proxmino (addresses: $PROXMINO_HOSTS)"
$DC up -d --build

if [[ "$MODEL" == *.gguf ]]; then  # your own trained model, for example models/proxmino-8b-q4_k_m.gguf
  echo "Loading your model file $MODEL"
  NAME=$(basename "$MODEL" | sed -E 's/-q4_k_m\.gguf$|\.gguf$//')
  CID=$($DC ps -q ollama)
  docker cp "$MODEL" "$CID:/root/.ollama/$NAME.gguf"
  MF=$(dirname "$MODEL")/Modelfile.$NAME
  if [ -f "$MF" ]; then sed "s|^FROM .*|FROM /root/.ollama/$NAME.gguf|" "$MF" | $DC exec -T ollama sh -c "cat > /root/.ollama/Modelfile.$NAME"
  else echo "FROM /root/.ollama/$NAME.gguf" | $DC exec -T ollama sh -c "cat > /root/.ollama/Modelfile.$NAME"; fi
  $DC exec -T ollama ollama create "$NAME" -f "/root/.ollama/Modelfile.$NAME"
  MODEL=$NAME
else
  echo "Downloading the local model $MODEL (a few GB, once)"
  $DC exec -T ollama ollama pull "$MODEL"
fi

if [ -f .env ]; then  # keys from the copied folder go into the data volume, readable by the app only
  echo "Moving your keys from .env into the app's data volume"
  $DC exec -T proxmino sh -c 'cat > /data/.env && chmod 600 /data/.env' < .env
fi

echo "Turning on local models (and cloud models if a key is set)"
$DC exec -T -e MODEL="$MODEL" proxmino python - <<'PY'
import os
from proxmino.config import Config
c = Config()
cloud = any(os.environ.get(k) for k in ("DEEPSEEK_API_KEY", "OLLAMA_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"))
tiers = {t: (chain if f"ollama:{os.environ['MODEL']}" in chain else chain + [f"ollama:{os.environ['MODEL']}"])
         for t, chain in (c.get("tiers") or {}).items()}
c.save_settings({"defaults": {"allow_local": True, "allow_cloud": cloud}, "tiers": tiers})
print("cloud models:", "on" if cloud else "off (add a key in Settings > API keys)")
PY
$DC restart proxmino >/dev/null

if ! $DC exec -T proxmino test -s /data/auth.json; then
  echo "Choose the password you will sign in with from other devices:"
  $DC exec proxmino python -m proxmino password
fi
echo
echo "Done. Open http://${IP:-$HOST.local}:3000 on your phone or Mac and sign in."
