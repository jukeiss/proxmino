## Download

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1/M2/M3/M4) | `Proxmino-…-mac-arm64.dmg` |
| Mac with Intel | `Proxmino-…-mac-x64.dmg` |
| Windows | `Proxmino-…-Setup-x64.exe` (installer) or `Proxmino-…-portable-x64.exe` (no install) |
| Ubuntu, Debian, Mint, Pop!_OS | `Proxmino-…-linux-x64.deb` |
| Fedora, RHEL, openSUSE | `Proxmino-…-linux-x64.rpm` |
| Any Linux | `Proxmino-…-linux-x64.AppImage` or `Proxmino-…-linux-x64.flatpak` |

**First launch.** These builds aren't code-signed yet, so your system will ask once:
- **macOS:** open the .dmg, drag Proxmino to Applications, then right-click Proxmino → **Open** → **Open**. On recent macOS: System Settings → Privacy & Security → **Open Anyway**.
- **Windows:** if SmartScreen appears, click **More info** → **Run anyway**.
- **AppImage:** `chmod +x Proxmino-*.AppImage && ./Proxmino-*.AppImage`
- **Flatpak:** `flatpak install --user Proxmino-*.flatpak`, then `flatpak run io.github.jukeiss.Proxmino`
- **.deb:** `sudo apt install ./Proxmino-*.deb` · **.rpm:** `sudo dnf install ./Proxmino-*.rpm`

Proxmino opens in your browser at `http://127.0.0.1:3000` and shows a small window to reopen it or quit. It needs [Ollama](https://ollama.com) running with a model (see the quick start guide). Your data stays in `~/.proxmino`.

Verify a download with `SHA256SUMS.txt`.
