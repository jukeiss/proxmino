# 1. Quick start

Proxmino runs on your own computer. Nothing you type leaves it unless you turn on web search.

## What you need

- macOS or Linux (Windows works through WSL)
- Python 3.10 or newer
- [Ollama](https://ollama.com), the program that runs the model
- 8 GB of RAM for **Proxmino Lite**, 16 GB for **Proxmino v5**

## Steps

1. Get the code:

   ```bash
   git clone https://github.com/jukeiss/proxmino
   cd proxmino
   ```

2. Install Ollama and start it:

   ```bash
   brew install ollama   # Linux: curl -fsSL https://ollama.com/install.sh | sh
   ollama serve
   ```

3. Add a Proxmino model ([2. Models](02-models.md) shows how). Lite is the quickest start:

   ```bash
   ./scripts/get-model.sh lite
   ```

4. Start the app:

   ```bash
   ./run.sh
   ```

   The first run creates `.venv` and installs about six small packages. Then open http://127.0.0.1:3000.

## Check it works

Ask it "What is 17 * 23 + 144 / 12?". It should answer 403 and show that it checked the sum with a small
program. If the reply says "demo mode", Ollama is not running or no model is installed: go back to step 2.

## Next

- [2. Models](02-models.md): which model to pick
- [3. Using the app](03-using-the-app.md): search, think mode, files, agents
