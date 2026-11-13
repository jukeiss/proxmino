#!/usr/bin/env bash
# Train Proxmino on a free Google Colab T4 from the Mac, and survive dropped connections:
# training runs detached on the Colab machine; the Mac checks in every 5 minutes, copies each checkpoint home,
# and if Google takes the machine back it gets a new one and resumes from the last copied checkpoint.
# Usage: run_v5_colab.sh <name> <base>. Ends when the adapter is copied home; the model file is built separately.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_v5_colab.sh <name> <hf base id>}; BASE=${2:?}
C="$HOME/.venvs/colab-cli/bin/colab --auth oauth2"; S=$NAME; LOG=logs/colab-$NAME.log; B=training/ckpt-$NAME
mkdir -p $B logs
say() { echo "$(date '+%H:%M') $*" | tee -a "$LOG"; }
py() { echo "$1" | perl -e 'alarm 150; exec @ARGV' $C exec -s $S 2>/dev/null; }  # a hung call gives up after 150 s   # run a little Python on the Colab machine

start_vm() {
  perl -e "alarm 120; exec @ARGV" $C sessions > /tmp/colab-sessions.txt || { say "could not reach Colab (network or sign-in); trying again in 10 min"; return 1; }
  if grep -q "\[$S\]" /tmp/colab-sessions.txt; then  # one failed call is not a lost machine: try three times
    for i in 1 2 3; do py "print('alive')" | grep -q alive && return 0; sleep 30; done
  fi
  $C stop -s $S >/dev/null 2>&1
  $C new -s $S --gpu ${GPU:-T4} || { say "no free ${GPU:-T4} right now"; return 1; }
  DATA_ZIP=${DATA_ZIP:-proxmino-data.zip}; [ -f "$DATA_ZIP" ] || zip -qr "$DATA_ZIP" data_big
  $C upload -s $S "$DATA_ZIP" /content/proxmino-data.zip >/dev/null
  $C upload -s $S training/train_colab_v5.py /content/train_colab_v5.py >/dev/null
  last=$(ls -d $B/checkpoint-*/ 2>/dev/null | while read d; do [ -f "$d/.complete" ] && echo "$d"; done | sort -t- -k2 -n | tail -1)
  if [ -n "$last" ]; then
    n=$(basename "$last"); say "resuming from $n"
    py "import os; os.makedirs('/content/ckpt/$n', exist_ok=True)" >/dev/null
    for f in "$last"*; do  # the upload API drops big files, so files over 30 MB go up in pieces and are joined there
      [ -f "$f" ] || continue; b=$(basename "$f")
      if [ $(stat -f%z "$f") -gt 31000000 ]; then
        tmp=$(mktemp -d); split -b 30m "$f" "$tmp/p_"
        for p in "$tmp"/p_*; do $C upload -s $S "$p" "/content/ckpt/$(basename "$p")" >/dev/null || say "upload of $b failed"; done
        py "import glob,os; P=sorted(glob.glob('/content/ckpt/p_*')); open('/content/ckpt/$n/$b','wb').write(b''.join(open(p,'rb').read() for p in P)); [os.remove(p) for p in P]" >/dev/null
        rm -rf "$tmp"
      else
        $C upload -s $S "$f" "/content/ckpt/$n/$b" >/dev/null || say "upload of $b failed"
      fi
    done
  fi
  py "import os, subprocess; print('already running') if os.popen('pgrep -f train_colab_v5[.]py').read().strip() else (subprocess.Popen('PROXMINO_BASE=$BASE PROXMINO_NAME=$NAME nohup python /content/train_colab_v5.py > /content/run.out 2>&1 &', shell=True), print('started'))"
}

backup() {
  last=$($C ls -s $S /content/ckpt 2>/dev/null | grep -oE "checkpoint-[0-9]+" | sort -t- -k2 -n | tail -1)
  [ -z "$last" ] || [ -f "$B/$last/.complete" ] && return
  mkdir -p "$B/$last"; ok=1
  for f in $($C ls -s $S /content/ckpt/$last 2>/dev/null | grep -oE "[A-Za-z0-9_.-]+\.(safetensors|json|pt|pth|bin|jinja)" | sort -u); do
    $C download -s $S /content/ckpt/$last/$f "$B/$last/$f" >/dev/null 2>&1 || ok=0
  done
  [ $ok = 1 ] && touch "$B/$last/.complete" && say "copied $last to the Mac"
}

caffeinate -is -w $$ &
tries=0
until [ $tries -ge 36 ]; do
  start_vm || { tries=$((tries+1)); sleep 600; continue; }
  while true; do
    sleep 300
    line=$(py "import os; L=open('/content/train_$NAME.log').read().splitlines() if os.path.exists('/content/train_$NAME.log') else ['']; print(L[-1]); print('RUNNING' if os.popen('pgrep -f train_colab_v5[.]py').read().strip() else 'STOPPED')")
    if [ -z "$line" ]; then sleep 60; line=$(py "import os; L=open('/content/train_$NAME.log').read().splitlines() if os.path.exists('/content/train_$NAME.log') else ['']; print(L[-1]); print('RUNNING' if os.popen('pgrep -f train_colab_v5[.]py').read().strip() else 'STOPPED')"); fi
    echo "$line" | tail -1 | grep -qE "^(RUNNING|STOPPED)$" || line=""  # an error message from the tool is not a status
    if [ -z "$line" ]; then say "lost the Colab machine; getting a new one"; tries=$((tries+1)); break; fi
    say "$(echo "$line" | head -1 | cut -c1-160)"
    backup
    if [ ! -f training/$NAME-adapter/.complete ] && py "import os; print('Y' if os.path.exists('/content/result/adapter/adapter_model.safetensors') and 'ADAPTER READY' in open('/content/train_$NAME.log').read() else 'N')" | grep -q "^Y"; then
      mkdir -p training/$NAME-adapter; ok=1
      for f in $($C ls -s $S /content/result/adapter 2>/dev/null | grep -oE "[A-Za-z0-9_.-]+\.(safetensors|json|jinja|txt|model)" | sort -u); do
        $C download -s $S /content/result/adapter/$f training/$NAME-adapter/$f >/dev/null 2>&1 || ok=0
      done
      [ $ok = 1 ] && [ -f training/$NAME-adapter/adapter_model.safetensors ] && touch training/$NAME-adapter/.complete && py "open('/content/result/adapter_fetched','w').write('1'); print('ok')" >/dev/null && say "copied the finished $NAME adapter to the Mac (training/$NAME-adapter)"
    fi
    if echo "$line" | grep -q "ALL DONE" && [ -f training/$NAME-adapter/.complete ]; then
      say "$NAME TRAINED; adapter in training/$NAME-adapter"; $C stop -s $S >/dev/null 2>&1; exit 0
    fi
    if echo "$line" | grep -q STOPPED; then
      say "training stopped on Colab; last output:"; py "print(open('/content/run.out').read()[-1500:])" | tee -a "$LOG"; exit 2
    fi
  done
done
say "gave up after $tries tries"; exit 1
