# 2. Models

| Model | Base | File | RAM | Best for |
|---|---|---|---|---|
| **Proxmino v5** | IBM Granite 4.2 8B | about 5 GB | 16 GB | Best answers, coding and maths |
| **Proxmino Lite** | IBM Granite 4.2 3B | about 2 GB | 8 GB | Older or smaller computers, faster replies |

Both are Q4_K_M GGUF files (about 4.5 bits per weight) and run with [Ollama](https://ollama.com).

> **Status:** Proxmino v5 is released
> ([proxmino-v5-1.0](https://github.com/jukeiss/proxmino/releases/tag/proxmino-v5-1.0)). It was trained for
> 1,073 steps on a free Colab T4; its held-out loss went from 0.869 at step 200 to 0.800 at the end. Proxmino
> Lite is still training. Benchmark scores for both, against plain Granite 4.2, will be added here when measured.

## Add a model

```bash
./scripts/get-model.sh lite    # or: v5
```

The script downloads the release files, joins the parts (GitHub allows 2 GB per file), checks the SHA-256
checksum, and adds the model to Ollama as `proxmino:lite` or `proxmino:v5`. The app uses v5 when it's
installed and Lite otherwise (see `tiers` in `config.yaml`).

## Use any other model

Any model in Ollama works. Pull one and the app finds it:

```bash
ollama pull <model-name>   # any model from https://ollama.com/library
```

To choose the order yourself, edit the `tiers` lists in `config.yaml`.
