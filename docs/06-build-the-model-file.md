# 6. Build the model file (GGUF)

Training gives you an adapter, a small add-on to Granite. Ollama needs one file that holds the whole model.
`training/build_gguf.py` makes that file. It needs no GPU, so a free CPU-only Colab machine (or any Linux
computer with about 40 GB of free disk) works.

It does four things with [llama.cpp](https://github.com/ggml-org/llama.cpp):

1. Downloads the Granite base and converts it to GGUF.
2. Converts your adapter to GGUF.
3. Merges the two (`llama-export-lora`).
4. Shrinks the result to Q4_K_M (`llama-quantize`): about 4.5 bits per weight, so an 8B model is about 5 GB
   and a 3B about 2 GB, with very little loss in quality.

## Run it

Upload your adapter folder as `/content/adapter` and the script, then:

```bash
GGUF_BASE=ibm-granite/granite-4.2-8b GGUF_ADAPTER=/content/adapter GGUF_NAME=my-proxmino python build_gguf.py
```

It takes about 50 minutes on a free 2-CPU machine. The result is `/content/result/my-proxmino-q4_k_m.gguf`.
Add `GGUF_KEEP_BASE=1` to also get the plain Granite file, for comparing the two.

## Getting a big file off Colab

Colab's file download can fail on files over about 80 MB. Copying over SSH works:

```bash
colab ssh -s my-session      # once, to check you can connect
scp -o "ProxyCommand=colab ssh --proxy-mode -s my-session" root@colab:/content/result/my-proxmino-q4_k_m.gguf .
```

## Add it to Ollama

```bash
cp my-proxmino-q4_k_m.gguf models/
sed "s#^FROM .*#FROM ./my-proxmino-q4_k_m.gguf#" models/Modelfile.template > models/Modelfile.mine
cd models && ollama create my-proxmino -f Modelfile.mine
```

Then put `ollama:my-proxmino` first in the model lists in `config.yaml`.
