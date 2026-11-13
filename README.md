# Proxmino

**An AI assistant that runs on your own computer.** It searches the web with citations, does deep research,
checks facts, shows its reasoning, writes and tests code, reads your PDFs and spreadsheets, and has a crew of
agents with their own personalities. No account, no API key, and nothing leaves your machine unless you turn
on web search.

Proxmino comes with its own open models, fine-tuned from IBM Granite 4.2:

| Model | Base | File size | Needs |
|---|---|---|---|
| **Proxmino v5** | Granite 4.2 8B | about 5 GB | 16 GB RAM |
| **Proxmino Lite** | Granite 4.2 3B | about 2 GB | 8 GB RAM |

> The v5 and Lite models are training now. Their files and test results will be added to this page and to the
> [Releases](https://github.com/jukeiss/proxmino/releases) when they are done. Until then the app works with
> any model you have in Ollama (it picks one automatically).

## Quick start

```bash
git clone https://github.com/jukeiss/proxmino
cd proxmino
./run.sh
```

Then open http://127.0.0.1:3000. You also need [Ollama](https://ollama.com) running with a model; the
[quick start guide](docs/01-quickstart.md) walks through it.

## Tutorials

1. [Quick start](docs/01-quickstart.md)
2. [Models](docs/02-models.md): which one to pick and how to add it
3. [Using the app](docs/03-using-the-app.md): Auto mode, Think, Search, DeepSearch, files, agents
4. [The coding agent](docs/04-coding-agent.md)
5. [Fine-tune your own Proxmino on a free Colab GPU](docs/05-fine-tune-on-colab.md)
6. [Build the model file (GGUF)](docs/06-build-the-model-file.md)
7. [Test your model](docs/07-test-your-model.md)
8. [Host it on a home server (CasaOS)](docs/self-host-casaos.md)

## What it does

| Feature | How to use it |
|---|---|
| Chat with streaming, stop, regenerate, edit, branches | Composer; hover a message for actions; arrows switch branches |
| Web search with inline `[n]` citations and confidence | `Search` pill, `/search`, or automatic (Auto mode) for recent events and for factual questions about a specific name |
| Auto mode picks the approach | Searches for recent or named facts, thinks step by step for maths, logic and debugging, uses tools when you ask it to make a chart, table or page; the reason shows as the first step |
| DeepSearch: plan, parallel search, gap-filling round, report, verification | `DeepSearch` pill or `/deep` |
| Fact-check: claims, then search, then verdicts, then Warden verification | `/check <claim or URL>`, or `/check` alone to check the last answer |
| Think mode: reasoning shown in a collapsible block | `Think` pill or `/think` |
| Heavy mode: Atlas, Probe and Edge draft in parallel, then Nexus merges | `Heavy` pill or `/heavy` |
| Agents with persona cards | `@forge`, `@probe`, `@rogue`... or the Agent picker |
| Tones | `/fun /blunt /formal /eli5 /analyst /coach /standard /professional` |
| Explain or summarize text or URLs | `/explain`, `/tldr`, or paste a bare URL |
| Memory | `/remember`, `/memory`, `/forget`, or the Memory panel |
| Code runner | `Run` button on Python blocks, or `/run` |
| Reply language | `/lang es <message>` |
| Your own bots | Bots > New bot: name, color, body shape, role, instructions, model, creativity. Saved as YAML in `~/.proxmino/agents/` |
| Group chats | Bots > New group: pick two or more bots; the first leads, `@name` picks who answers, `@all` asks everyone, bots hand off with `@id` |
| Bot marketplace | Marketplace in the sidebar: 15 ready-made bots in 7 categories; Get adds one, then edit it like your own |
| Your own models and keys | `config.yaml` providers and tiers; keys in `.env`; Settings panel |

## Privacy and safety

- Your chats, memory and settings stay in `data/` on your computer. Delete that folder to reset.
- Cloud models (DeepSeek and others) are off until you add a key and switch them on in Settings.
- **Security for a local app that can run code:**
  - The API only answers requests addressed to localhost and requires a custom header, which blocks other websites and DNS-rebinding attacks.
  - Code runs under the macOS sandbox with no network, no writes outside a temp folder, no reads from your home folder, plus CPU and process limits.
  - Fetched web pages are wrapped as untrusted data, and page fetches refuse private or loopback addresses.
- Posts, emails and messages are never sent without your click.

## Layout

```
proxmino/
  config.py     config.yaml + data/settings.json + .env
  db.py         SQLite schema, branching message tree, memory FTS, search cache
  llm.py        Ollama and OpenAI-compatible providers, router, fallback, think-tag parsing
  search.py     ddgs / SearXNG, page fetch and extraction, BM25 ranking
  prompts.py    fixed prefix, tones, agent cards, task prompts
  pipeline.py   routing, slash commands, handlers (chat, search, think, check, deep, heavy, run)
  safety.py     emoji filter, citation verifier, confidence
  sandbox.py    Python runner
  server.py     FastAPI + SSE
crew/           the nine built-in agents (YAML)
web/index.html  the whole UI
tests/          offline tests (fake model and fake search)
```

Data (chats, memory, settings) lives in `data/`. Delete that folder to reset.

## Contributing

Pull requests are welcome. To change something, fork the repository, make your change on a branch, and open a
pull request; `main` only changes through reviewed pull requests. Run `./run.sh test` first (all tests run
offline with a fake model).

## Licence

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 jukeiss.

The models are fine-tuned from [IBM Granite](https://huggingface.co/ibm-granite) (Apache-2.0). Training data and
their licences are listed in [MODEL_AND_DATA_LICENSES.md](MODEL_AND_DATA_LICENSES.md). Some parts are adapted from
[Rakazo](https://github.com/elie222/rakazo) (Apache-2.0); see [third_party/rakazo](third_party/rakazo/NOTICE.md).
