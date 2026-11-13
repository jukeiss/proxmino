#!/usr/bin/env bash
# Download a Proxmino model from the GitHub release, check it, and add it to Ollama.
#   ./scripts/get-model.sh lite     # Proxmino Lite (3B, about 2 GB)
#   ./scripts/get-model.sh v5       # Proxmino v5 (8B, about 5 GB)
# GitHub limits release files to 2 GB, so big models come in parts that are joined here.
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH=${1:-lite}
case "$WHICH" in
  lite) FILE=proxmino-lite-q4_k_m.gguf; TAG=proxmino-lite-1.0; NAME=proxmino:lite ;;
  v5)   FILE=proxmino-v5-q4_k_m.gguf;   TAG=proxmino-v5-1.0;   NAME=proxmino:v5 ;;
  *) echo "usage: $0 lite|v5"; exit 1 ;;
esac
REPO=jukeiss/proxmino
mkdir -p models
command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com"; exit 1; }
if [ ! -f "models/$FILE" ]; then
  echo "Finding the files for $TAG..."
  urls=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$TAG" |
         python3 -c "import json,sys; [print(a['browser_download_url']) for a in json.load(sys.stdin)['assets']]")
  tmp=$(mktemp -d)
  for u in $urls; do
    echo "Downloading $(basename "$u")"
    curl -fL --retry 5 -C - -o "$tmp/$(basename "$u")" "$u"
  done
  cat "$tmp/$FILE".part-* > "models/$FILE" 2>/dev/null || mv "$tmp/$FILE" "models/$FILE"
  want=$(grep " $FILE\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
  got=$( (shasum -a 256 "models/$FILE" 2>/dev/null || sha256sum "models/$FILE") | cut -d' ' -f1)
  rm -rf "$tmp"
  [ "$want" = "$got" ] || { echo "Checksum does not match: the download is damaged. Delete models/$FILE and try again."; exit 1; }
  echo "Checksum OK"
fi
sed "s#^FROM .*#FROM ./$FILE#" models/Modelfile.template > "models/Modelfile.$WHICH"
(cd models && ollama create "$NAME" -f "Modelfile.$WHICH")
echo "Done. Ollama now has $NAME. Start the app with ./run.sh"
