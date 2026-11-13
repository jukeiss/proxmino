"""Merge a Proxmino LoRA adapter into its Granite base and make a Q4_K_M GGUF, on any free Colab machine (no GPU
needed). Uses llama.cpp tools only, streaming tensors, so it fits in 12 GB of RAM.
Env: GGUF_BASE (Hugging Face id, e.g. ibm-granite/granite-4.2-8b), GGUF_ADAPTER (folder with adapter_model.safetensors),
GGUF_NAME (e.g. proxmino-v5). Also makes the plain base GGUF when GGUF_KEEP_BASE=1, for fair comparisons.
Progress: /content/gguf_<name>.log. Result: /content/result/<name>-q4_k_m.gguf"""
import os, shutil, subprocess, sys, time

BASE_ID = os.environ.get("GGUF_BASE", "ibm-granite/granite-4.2-8b")
ADAPTER = os.environ.get("GGUF_ADAPTER", "/content/adapter")
NAME = os.environ.get("GGUF_NAME", "proxmino-v5")
KEEP_BASE = os.environ.get("GGUF_KEEP_BASE") == "1"
LOG = open(f"/content/gguf_{NAME}.log", "a", buffering=1)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


def sh(cmd):
    log("$", cmd[:160])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        log("FAILED:", (r.stderr or r.stdout)[-800:].replace("\n", " | "))
        raise SystemExit(1)
    return r.stdout


os.makedirs("/content/result", exist_ok=True)
log("disk free GB:", shutil.disk_usage("/content").free // 10**9, "| cpus:", os.cpu_count())
if not os.path.isdir("/content/llama.cpp"):
    sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")
sh(f"{sys.executable} -m pip install -q gguf sentencepiece 'transformers>=4.56' safetensors huggingface_hub peft")
if not os.path.exists("/content/llama.cpp/build/bin/llama-quantize"):
    sh("cd /content/llama.cpp && cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF > /dev/null && "
       f"cmake --build build --config Release -j {os.cpu_count()} --target llama-quantize llama-export-lora > /dev/null")
log("llama.cpp tools built")

base = "/content/base-" + BASE_ID.split("/")[-1]
if not os.path.exists(f"{base}/config.json"):
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_ID, local_dir=base, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
log("base model downloaded")

f16 = f"/content/{BASE_ID.split('/')[-1]}-f16.gguf"
if not os.path.exists(f16):
    sh(f"{sys.executable} /content/llama.cpp/convert_hf_to_gguf.py {base} --outtype f16 --outfile {f16}")
lora = f"/content/{NAME}-lora.gguf"
if not os.path.exists(lora):
    sh(f"{sys.executable} /content/llama.cpp/convert_lora_to_gguf.py {ADAPTER} --base {base} --outtype f16 --outfile {lora}")
shutil.rmtree(base, ignore_errors=True)  # 16 GB no longer needed
log("converted base and adapter")

merged = f"/content/{NAME}-f16.gguf"
if not os.path.exists(merged):
    sh(f"/content/llama.cpp/build/bin/llama-export-lora -m {f16} --lora {lora} -o {merged} -t {os.cpu_count()}")
if KEEP_BASE:
    sh(f"/content/llama.cpp/build/bin/llama-quantize {f16} /content/result/{BASE_ID.split('/')[-1]}-q4_k_m.gguf Q4_K_M {os.cpu_count()}")
os.remove(f16)
log("adapter merged")

out = f"/content/result/{NAME}-q4_k_m.gguf"
sh(f"/content/llama.cpp/build/bin/llama-quantize {merged} {out} Q4_K_M {os.cpu_count()}")
os.remove(merged)
log("ALL DONE", round(os.path.getsize(out) / 1e9, 2), "GB")
