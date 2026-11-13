# 5. Fine-tune your own Proxmino on a free Colab GPU

This is how Proxmino v5 and Proxmino Lite were trained, on Google Colab's free T4 GPU. You need a Google
account. Nothing here costs money.

## How it works

- **LoRA, not full training.** The Granite base model stays frozen. A small add-on (about 80 MB) learns the
  changes. Only the last 16 of the 40 layers get one, which keeps it fast and stops it from forgetting what
  Granite already knows.
- **4-bit loading** ([Unsloth](https://github.com/unslothai/unsloth)) lets an 8B model train in the T4's 15 GB.
- **Only the answers are learned.** The questions are masked, so the model learns how to reply, not how to
  ask.
- **Checkpoints every 200 steps.** Free Colab machines can be taken back at any time; training resumes from
  the last checkpoint.

## Data

`training/train_colab_v5.py` mixes:

| Part | Size | Licence |
|---|---|---|
| Your own conversations (`data_big/train.jsonl`) | 5,200 | yours |
| MATH train: step-by-step solutions | about 1,600 | MIT |
| MBPP train and validation: Python that passes tests | about 970 | CC-BY-4.0 |
| WebSight: clean HTML pages | 900 | CC-BY-4.0 |
| Cosmopedia khanacademy and openstax: teaching texts | 1,400 | Apache-2.0 |

Test splits are never used, so the scores in the README are fair.

Your own conversations go in `data_big/train.jsonl` and `data_big/valid.jsonl`, one per line:

```json
{"messages": [{"role": "system", "content": "You are Proxmino..."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

Zip the folder as `proxmino-data.zip`.

## Option A: in the browser

1. Open https://colab.research.google.com, choose **File > New notebook**, then **Runtime > Change runtime
   type > T4 GPU**.
2. Upload `proxmino-data.zip` and `training/train_colab_v5.py` with the folder icon on the left.
3. Run this in a cell (use `granite-4.2-3b` for a Lite model):

   ```python
   !PROXMINO_BASE=ibm-granite/granite-4.2-8b PROXMINO_NAME=my-proxmino python train_colab_v5.py
   ```

4. When it prints `ALL DONE`, download the `result/adapter` folder.

An 8B model takes about 3 hours on a T4; the 3B takes about 1. Keep the tab open.

## Option B: from your terminal (survives dropped machines)

With the [Colab CLI](https://pypi.org/project/google-colab-cli/) signed in:

```bash
./training/run_v5_colab.sh my-proxmino ibm-granite/granite-4.2-8b
```

It starts a T4, uploads everything, watches the run every 5 minutes, copies each checkpoint to your
computer, and gets a new machine and resumes if Google takes the old one back. The adapter lands in
`training/my-proxmino-adapter`.

## Free GPU limits

Free Colab gives a few hours of T4 a day. When the limit is reached, Colab answers "Service Unavailable" for
a while (it was about a day for us). The runner keeps trying every 10 minutes.

Next: [6. Build the model file](06-build-the-model-file.md).
