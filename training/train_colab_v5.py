"""Proxmino v5 / Proxmino Lite on a free Colab T4. Base from PROXMINO_BASE (ibm-granite/granite-4.2-8b or -3b, Apache-2.0),
name from PROXMINO_NAME. Data: part of Proxmino's own chats (data_big), step-by-step maths (MATH train, MIT), tested Python
(MBPP train+validation, CC-BY-4.0), clean web pages (WebSight, CC-BY-4.0) and teaching texts (Cosmopedia khanacademy and
openstax subsets, Apache-2.0; synthetic, not Khan Academy's own material). Test splits are never used.
Progress: /content/train_<name>.log. Result: /content/result/adapter. Resumes from /content/ckpt."""
import glob, inspect, json, os, shutil, subprocess, sys, time, zipfile

BASE = os.environ.get("PROXMINO_BASE", "ibm-granite/granite-4.2-8b")
NAME = os.environ.get("PROXMINO_NAME", "proxmino-v5")
LOG = open(f"/content/train_{NAME}.log", "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n")

if not os.path.isdir("/content/data_big"):
    zipfile.ZipFile("/content/proxmino-data.zip").extractall("/content")
try:
    import unsloth  # noqa: F401
except ImportError:
    log("installing unsloth"); subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

log("gpu:", torch.cuda.get_device_name(0))
MAX_LEN, CKPT, OUT = 2048, "/content/ckpt", "/content/result"
os.makedirs(OUT, exist_ok=True)
model, tokenizer = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True, dtype=None)
N = model.config.num_hidden_layers
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=160, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=list(range(N - 16, N)), use_gradient_checkpointing="unsloth", random_state=0)

def load(name):
    texts, long = [], 0
    for line in open(f"/content/data_big/{name}.jsonl"):
        t = tokenizer.apply_chat_template(json.loads(line)["messages"], tokenize=False)
        if len(tokenizer(t, add_special_tokens=False)["input_ids"]) > MAX_LEN:
            long += 1; continue
        texts.append(t)
    log(f"{name}: {len(texts):,} conversations ({long} too long, left out)")
    return Dataset.from_dict({"text": texts})
import random, re
from datasets import load_dataset
SYSTEM = "You are Proxmino, a helpful assistant. Be accurate and clear; say so when you are not sure."
random.seed(0)

def fits(msgs):
    t = tokenizer.apply_chat_template(msgs, tokenize=False)
    return t if len(tokenizer(t, add_special_tokens=False)["input_ids"]) <= MAX_LEN else None

own = [json.loads(l)["messages"] for l in open("/content/data_big/train.jsonl")]
random.shuffle(own)
convos = own[:5200]  # Proxmino's own voice and app skills
extra = {"math": 0, "code": 0, "web": 0}
for sub in ["algebra", "counting_and_probability", "number_theory", "prealgebra", "intermediate_algebra", "geometry", "precalculus"]:
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", sub, split="train").shuffle(seed=2)
    except Exception as e:
        log("MATH", sub, "not loaded:", str(e)[:120]); continue
    for r in ds.select(range(min(230, len(ds)))):
        m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r["solution"])
        if not m:
            continue
        convos.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["problem"]},
                       {"role": "assistant", "content": r["solution"].strip() + f"\n\nFinal answer: {m[-1].strip()}"}])
        extra["math"] += 1
for split in ("train", "validation"):
    for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
        code = r["code"].replace("\r\n", "\n").strip()
        if r.get("test_setup_code"):
            code = r["test_setup_code"].strip() + "\n" + code
        tests = "\n".join(r["test_list"])
        convos.append([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": r["text"] + "\nYour code should pass these tests:\n" + tests},
                       {"role": "assistant", "content": f"```python\n{code}\n```\n\nIt passes the given tests:\n```python\n{tests}\n```"}])
        extra["code"] += 1
try:
    web = load_dataset("HuggingFaceM4/WebSight", data_files="data/train-00000-of-00071-eb722b04b83e13b7.parquet",
                       split="train", columns=["text", "llm_generated_idea"]).shuffle(seed=0).select(range(1500))
    for r in web:
        idea, html = (r.get("llm_generated_idea") or "").strip(), (r.get("text") or "").strip()
        if idea and html and len(html) < 6000 and extra["web"] < 900:
            convos.append([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"Design a clean, modern web page: {idea}. Give the complete HTML in one ```html block."},
                           {"role": "assistant", "content": f"```html\n{html}\n```"}])
            extra["web"] += 1
except Exception as e:
    log("WebSight not loaded:", str(e)[:150])
for sub, cap in (("khanacademy", 700), ("openstax", 700)):  # teaching texts: explain a topic clearly, step by step
    try:
        n = 0
        for r in load_dataset("HuggingFaceTB/cosmopedia", sub, split="train", streaming=True):
            p, t = (r.get("prompt") or ""), (r.get("text") or "").strip()
            title = re.search(r"titled ['\"]([^'\"]{3,120})['\"]", p)
            course = re.search(r"textbook on ['\"]([^'\"]{3,120})['\"]", p)
            if title and course and t and len(t) < 5000:
                convos.append([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": f"Teach me \"{title.group(1)}\" ({course.group(1)}), clearly and step by step."},
                               {"role": "assistant", "content": t}])
                n += 1
            if n >= cap:
                break
        extra[sub] = n
    except Exception as e:
        log("Cosmopedia", sub, "not loaded:", str(e)[:150])
random.shuffle(convos)
texts = [t for t in (fits(c) for c in convos) if t]
log(NAME, "data:", len(texts), "conversations;", extra, "plus", 5200, "of Proxmino's own")
train_ds, valid_ds = Dataset.from_dict({"text": texts}), load("valid")

sig = inspect.signature(SFTConfig.__init__).parameters
kw = dict(output_dir=CKPT, dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
          num_train_epochs=1, learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=30, weight_decay=0.01,
          max_grad_norm=1.0, optim="adamw_8bit", fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
          logging_steps=10, save_steps=200, save_total_limit=3, eval_steps=200, per_device_eval_batch_size=2,
          load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
          seed=0, report_to="none", packing=False, disable_tqdm=True)
kw["max_length" if "max_length" in sig else "max_seq_length"] = MAX_LEN
kw["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "steps"
kw = {k: v for k, v in kw.items() if k in sig}

class Progress(TrainerCallback):
    def on_train_begin(self, args, state, control, **k):
        self.t0, self.s0 = time.time(), state.global_step
    def on_log(self, args, state, control, logs=None, **k):
        done = state.global_step - self.s0
        eta = (time.time() - self.t0) / max(done, 1) * (state.max_steps - state.global_step)
        log(f"step {state.global_step}/{state.max_steps}", {k2: round(v, 4) for k2, v in (logs or {}).items() if isinstance(v, float)},
            f"| about {eta/3600:.1f} h left")

tsig = inspect.signature(SFTTrainer.__init__).parameters
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=valid_ds, args=SFTConfig(**kw), callbacks=[Progress()],
                     **{("processing_class" if "processing_class" in tsig else "tokenizer"): tokenizer})
# Granite 4.1 uses <|start_of_role|> markers; Granite 4.2 uses ChatML (<|im_start|>). Learn only the assistant's words.
probe = tokenizer.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], tokenize=False)
if "<|im_start|>" in probe:
    parts = ("<|im_start|>user\n", "<|im_start|>assistant\n")
else:
    parts = ("<|start_of_role|>user<|end_of_role|>", "<|start_of_role|>assistant<|end_of_role|>")
assert parts[0] in probe and parts[1] in probe, probe
log("chat format:", repr(probe[-120:]))
trainer = train_on_responses_only(trainer, instruction_part=parts[0], response_part=parts[1])
saved = sorted(glob.glob(f"{CKPT}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
log("resuming from " + saved[-1] if saved else "starting fresh")
trainer.train(resume_from_checkpoint=saved[-1] if saved else None)
log("training done, best held-out loss:", trainer.state.best_metric)

model.save_pretrained(f"{OUT}/adapter"); tokenizer.save_pretrained(f"{OUT}/adapter")
log("ADAPTER READY")
log("ALL DONE")
