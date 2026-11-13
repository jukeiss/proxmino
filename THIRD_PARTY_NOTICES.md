# Third-party notices

Proxmino includes or uses the third-party software, models and data below. Their licenses allow
commercial use; the obligations column says what a copy you sell or give away must include.
Checked 2026-10-29 from each package's own metadata. Not legal advice.

## Code included in Proxmino

| Component | License | Obligation |
|---|---|---|
| Rakazo (parts adapted, see `third_party/rakazo/NOTICE.md`) | Apache-2.0 | Ship `third_party/rakazo/LICENSE` and `NOTICE.md`; keep the list of changes; do not use the Rakazo name or logo to promote Proxmino (Apache-2.0 grants no trademark rights) |

## Python packages (installed by `run.sh`, not copied into the source)

| Package | License |
|---|---|
| fastapi, pydantic, pydantic_core, annotated-types, annotated-doc, typing-inspection, anyio, h11, httptools, watchfiles, PyYAML, ddgs, primp, pluggy, iniconfig, pytest | MIT |
| uvicorn, starlette, httpx, httpcore, idna, click, lxml, python-dotenv, websockets | BSD-3-Clause |
| Pygments | BSD-2-Clause |
| pypdf (reads attached PDFs) | BSD-3-Clause |
| uvloop | MIT or Apache-2.0 |
| packaging, opentelemetry-api | Apache-2.0 (packaging: Apache-2.0 or BSD-2-Clause) |
| typing_extensions | PSF-2.0 |
| certifi | MPL-2.0 (used unmodified; if you ever modify its files, those files must stay MPL-2.0 and their source must be available) |

If you ship a bundle that contains these packages (for example a packaged app), include each package's
license text. A plain source copy that installs them with pip does not contain them.

## Models and training data

The current models and every dataset used to train them are listed, with what each requires, in
`MODEL_AND_DATA_LICENSES.md` (checked 2026-11-05). The table below is the older 2026-10-29 list.

## Models (downloaded by the user through Ollama, not bundled) (older list)

| Model | License | Note |
|---|---|---|
| phi4-mini (Microsoft) | MIT | Fine for commercial use |
| SmolLM3-3B (Hugging Face), base of praxis-1 | Apache-2.0 | praxis-1 is a derivative: ship the Apache-2.0 license and a notice if you distribute it |
| Proxmino 2 (Phi-4-mini-instruct + our LoRA fine-tune, `~/praxis-model/lm/finetune.py`) | MIT (from the base) | A derivative of Phi-4-mini: ship Microsoft's MIT license text and copyright notice with the model file; say it is based on Phi-4-mini, without implying Microsoft endorses it. If built on SmolLM3-3B instead: Apache-2.0 license and a notice |
| proxmino:v1 (Phi-4-mini + our LoRA) | MIT (from the base) | Ship Microsoft's MIT license text and notice with the model |
| Training data for proxmino:8b | OpenAssistant oasst2 (Apache-2.0); GSM8K (MIT); NVIDIA HelpSteer2 (CC-BY-4.0) | Credit: "Contains data from NVIDIA HelpSteer2 (CC-BY-4.0), OpenAssistant oasst2 (Apache-2.0) and GSM8K (MIT)" in the model card |
| proxmino:8b (IBM Granite 3.3 8B Instruct + our LoRA) | Apache-2.0 (from the base) | Ship the Apache-2.0 license and a notice that it is based on IBM Granite; no implied IBM endorsement |
| Proxmino LM (trained from scratch in `~/praxis-model/lm`) | yours | Training text includes FineWeb-Edu (ODC-By 1.0): credit the dataset when you distribute the model |

## Optional and external services

| Service | Status for a paid product |
|---|---|
| Brave Search API, Tavily API (official; the default search providers) | Each customer uses their own account and key and must follow that provider's terms. Proxmino does not cache their results by default |
| SearXNG (a search server the customer runs) | Optional; SearXNG is AGPL-3.0 software but runs as a separate program, Proxmino only calls it over HTTP |
| Web search through `ddgs` (DuckDuckGo) | **Off by default** (`search.allow_unofficial: false`). The library's code is MIT, but its README says "for educational purposes only" and it automates DuckDuckGo's website rather than an official API: keep it off in a version you sell |
| Ollama, llama.cpp (used, not bundled) | MIT |
| Discord (official bot API) and IMAP/SMTP email | Optional, off until keys are added. Customers use their own bot or mailbox and follow those terms; WhatsApp is not supported because only its Business API allows automation |
| Ollama cloud models (DeepSeek V4.1 Flash, V4 Pro: MIT weights) | Optional, off by default (`defaults.allow_cloud`). Prompts go to ollama.com, billed per token to the customer's own Ollama account; customers accept Ollama's terms |
| Cloud model providers (OpenAI, OpenRouter, Groq), off by default | Customers bring their own keys and accept those providers' terms |
| Fonts | None bundled (system fonts) |
| Icons and bot artwork | Icons drawn for Proxmino; bot shapes adapted from Rakazo (Apache-2.0, above) |
