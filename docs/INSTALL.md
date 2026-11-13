# Installing Proxmino

This guide covers macOS, Linux, Windows and Docker, then adding a model, checking it works, updating and
uninstalling. If something goes wrong, see [Troubleshooting](#troubleshooting) at the end.

## What you need

| | Minimum | Recommended |
|---|---|---|
| System | macOS 12+, Linux (64-bit), or Windows 10/11 with WSL2 | macOS on Apple Silicon, or Linux with an NVIDIA GPU |
| Memory (RAM) | 8 GB (Proxmino Lite) | 16 GB or more (Proxmino v5) |
| Free disk | 5 GB | 10 GB |
| Python | 3.10 or newer | 3.12 |
| Ollama | any recent version | latest |

No GPU is needed. A GPU or Apple Silicon makes answers much faster.

## Easiest: the installer

Download the installer for your computer from the
[latest release](https://github.com/jukeiss/proxmino/releases/latest) (`.dmg` for Mac, `Setup-x64.exe` or
`portable-x64.exe` for Windows, `.deb`, `.rpm`, `.AppImage` or `.flatpak` for Linux). You still need
[Ollama](https://ollama.com) with a model; then skip to [Choosing a model](#choosing-a-model).

The builds aren't code-signed yet: on macOS right-click the app and choose **Open** the first time; on Windows
click **More info**, then **Run anyway**.

The rest of this guide installs from source, which works everywhere and makes updating easy.

---

## macOS

1. **Install Homebrew** (skip if `brew --version` already works). Follow the one-line command on https://brew.sh.

2. **Install Python, Git and Ollama:**

   ```bash
   brew install python@3.12 git ollama
   ```

3. **Start Ollama** and leave this terminal window open (or install the Ollama app from https://ollama.com,
   which starts by itself):

   ```bash
   ollama serve
   ```

4. **In a new terminal window, get Proxmino:**

   ```bash
   git clone https://github.com/jukeiss/proxmino
   cd proxmino
   ```

5. **Add a model** (see [Choosing a model](#choosing-a-model)):

   ```bash
   ./scripts/get-model.sh lite
   ```

6. **Start the app:**

   ```bash
   ./run.sh
   ```

   The first start creates a private Python environment in `.venv` and installs about six small packages
   (FastAPI, Uvicorn, httpx, PyYAML, ddgs, pytest). This takes a minute. Later starts are instant.

7. Open **http://127.0.0.1:3000** in your browser.

---

## Linux (Ubuntu, Debian, Mint, Pop!_OS)

1. **Install Python, venv and Git:**

   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl
   ```

   On Fedora: `sudo dnf install -y python3 git curl`. On Arch: `sudo pacman -S python git curl`.

2. **Install Ollama** (it sets itself up as a background service):

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   Check it's running with `curl http://127.0.0.1:11434` (it should say "Ollama is running").

3. **Get Proxmino, add a model and start it:**

   ```bash
   git clone https://github.com/jukeiss/proxmino
   cd proxmino
   ./scripts/get-model.sh lite
   ./run.sh
   ```

4. Open **http://127.0.0.1:3000**.

With an NVIDIA GPU, install the NVIDIA driver first; Ollama uses the GPU automatically.

---

## Windows (with WSL2)

Proxmino runs inside WSL2, Windows' built-in Linux. Your browser on Windows can still open it.

1. Open **PowerShell as Administrator** and run:

   ```powershell
   wsl --install -d Ubuntu
   ```

   Restart when asked, then open **Ubuntu** from the Start menu and create a Linux user name and password.

2. In the Ubuntu window, follow the [Linux](#linux-ubuntu-debian-mint-pop_os) steps above.

3. Open **http://127.0.0.1:3000** in your normal Windows browser.

Tip: keep the project inside the Linux home folder (`~/proxmino`), not under `/mnt/c`, or it will be slow.

---

## Docker (servers, home labs)

Runs the app, Ollama and a private SearXNG search server together. Nothing but the app's port is exposed.

```bash
git clone https://github.com/jukeiss/proxmino
cd proxmino
docker compose -f deploy/docker-compose.yml up -d --build
```

Add a model inside the Ollama container (any Ollama model name works, or copy a Proxmino `.gguf` in):

```bash
docker compose -f deploy/docker-compose.yml exec ollama ollama pull <model-name>
```

To open it from other devices, set a password and tell the app its address:

```bash
docker compose -f deploy/docker-compose.yml exec proxmino python -m proxmino password
PROXMINO_HOSTS=192.168.1.50 docker compose -f deploy/docker-compose.yml up -d
```

Then open `http://<server address>:3000` and sign in. For a full home-server walkthrough, see
[self-host-casaos.md](self-host-casaos.md).

Only put it on the public internet behind HTTPS (for example a Tailscale or Cloudflare tunnel), never with
port 3000 open directly.

---

## Choosing a model

| Model | Command | Download | RAM | Notes |
|---|---|---|---|---|
| Proxmino Lite | `./scripts/get-model.sh lite` | about 2 GB | 8 GB | Faster; good on laptops |
| Proxmino v5 | `./scripts/get-model.sh v5` | about 5 GB | 16 GB | Better at coding and maths |
| Any Ollama model | `ollama pull <name>` | varies | varies | Found automatically |

`get-model.sh` downloads the files from the [Releases page](https://github.com/jukeiss/proxmino/releases),
joins the parts, checks the SHA-256 checksum so a damaged download is caught, and adds the model to Ollama.

Manual install, if you prefer: download the parts and `SHA256SUMS` from the release, then

```bash
cat proxmino-v5-q4_k_m.gguf.part-* > models/proxmino-v5-q4_k_m.gguf
shasum -a 256 -c SHA256SUMS --ignore-missing      # Linux: sha256sum -c SHA256SUMS --ignore-missing
sed "s#^FROM .*#FROM ./proxmino-v5-q4_k_m.gguf#" models/Modelfile.template > models/Modelfile.v5
cd models && ollama create proxmino:v5 -f Modelfile.v5
```

---

## Check it works

1. Ask: **What is 17 * 23 + 144 / 12?** It should answer **403** and say it checked the sum with a program.
2. Ask: **Write a Python function that checks if a word is a palindrome.** You should see the code, then a
   step saying the tests passed.
3. Run the built-in tests (offline, with a fake model; about 15 seconds):

   ```bash
   ./run.sh test
   ```

   You should see `194 passed`.

---

## Optional extras

- **Web search with an API key** (more reliable than the default): put `BRAVE_SEARCH_API_KEY=...` or
  `TAVILY_API_KEY=...` in a file named `.env` (copy `.env.example`), or use Settings > Search.
- **Cloud models** (DeepSeek, OpenRouter and others): add the key to `.env` and switch on cloud models in
  Settings. They are off by default so nothing leaves your computer.
- **Better web page reading:** `.venv/bin/pip install trafilatura`.
- **Terminal use:** `./run.sh ask "your question"` and `./run.sh code "your coding task"`.

---

## Updating

```bash
cd proxmino
git pull
./run.sh
```

Your chats, memory and settings in `data/` are kept. If `requirements.txt` changed, delete `.venv` once and
`./run.sh` rebuilds it.

## Uninstalling

```bash
rm -rf proxmino                 # the app and your chats (back up data/ first if you want them)
ollama rm proxmino:lite proxmino:v5
rm -rf ~/.proxmino              # your own bots
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| The reply says **demo mode** | Ollama isn't running or has no model. Run `ollama serve` and `ollama list`. |
| `Python 3.10 or newer is required` | Install a newer Python (`brew install python@3.12` or `sudo apt install python3`). |
| `Address already in use` / port 3000 busy | Start on another port: `./run.sh --port 3001`. |
| Answers are very slow | Use Proxmino Lite, close other heavy apps, and make sure only one model is loaded (`ollama ps`). |
| `out of memory` or the computer freezes | The model is too big for your RAM: switch to Lite. |
| `get-model.sh` says the checksum doesn't match | The download was damaged. Delete `models/*.gguf` and run it again. |
| `permission denied: ./run.sh` | Run `chmod +x run.sh scripts/*.sh`. |
| Web search finds nothing | Add a Brave or Tavily key, or set up SearXNG (Settings > Search). |
| Something else | Open an issue on GitHub with what you ran and what it printed. |
