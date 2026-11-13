# Host Proxmino on your ThinkPad (CasaOS)

Proxmino runs on the ThinkPad around the clock; you open it from your phone or Mac in a browser and sign in
with a password. It brings its own Ollama (the program that runs local AI models), so you install nothing
from the CasaOS App Store.

## What you need

| | Minimum (cloud models only) | Recommended (local model too) | For your own proxmino:8b |
|---|---|---|---|
| CPU | 2 cores, 64-bit Intel or AMD | 4 cores (T480: i5-8250U is fine) | 4 cores or more |
| RAM | 4 GB | 8 GB | 16 GB |
| Free disk | 10 GB | 20 GB | 30 GB |
| System | Ubuntu Server 24.04 LTS (or Debian 12) with CasaOS | same | same |
| Network | Wi-Fi works; a cable is steadier | cable | cable |

- **Cloud only** means DeepSeek (or another provider with a key) answers. The ThinkPad then only runs the
  app (under 0.5 GB of RAM) and replies are as fast as on your Mac.
- **Local model** means the ThinkPad answers by itself with no key and no internet. The T480 has no
  graphics card that models can use, so it runs on the CPU. As a rough guide: a 3-4B model such as phi4-mini
  writes a few words per second; an 8B model about half that. Fine for drafts and the inbox, slow for long
  reports. Measure it on yours: the answer footer shows the speed.
- Raspberry Pi and other ARM boards are not supported by this package yet (it is set up for amd64).

## Part 1: prepare the ThinkPad (once, about 45 minutes)

1. **Back up anything on the ThinkPad.** Installing Ubuntu Server erases the disk.
2. On your Mac, download **Ubuntu Server 24.04 LTS** from ubuntu.com/download/server and write it to a USB
   stick with balenaEtcher (balena.io/etcher).
3. On the ThinkPad: plug in the stick, power on and press **F12** for the boot menu, pick the USB stick.
   - If it does not show: press **F1** for the BIOS, Security > Secure Boot: Disabled (Ubuntu usually boots
     with it on, so try without changing it first).
4. Install Ubuntu Server with the defaults. When asked:
   - choose a **username and password** (you will type this password when logging in to the server; keep it),
   - tick **Install OpenSSH server** (lets your Mac connect to it),
   - skip the extra "snaps".
5. When it restarts, log in on the ThinkPad and run:
   ```bash
   sudo apt update && sudo apt upgrade -y
   curl -fsSL https://get.casaos.io | sudo bash
   ```
   At the end CasaOS prints an address like `http://192.168.1.50`. Write it down: that is the ThinkPad's
   address on your network.
6. Keep it awake with the lid closed:
   ```bash
   sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
   sudo systemctl restart systemd-logind
   ```
7. Open the address from step 5 in a browser on your Mac and create your CasaOS account.

Tip: in your router's settings, give the ThinkPad a fixed (reserved) address so it never changes.

## Part 2: install Proxmino (about 10 minutes, then the model download)

Easiest way, from your Mac (Terminal, in the folder that has `Proxmino-server.zip`):

```bash
scp Proxmino-server.zip you@192.168.1.50:
ssh you@192.168.1.50
```
Use your Ubuntu username and the ThinkPad's address. Then, now on the ThinkPad:

```bash
sudo apt install -y unzip
unzip Proxmino-server.zip -d proxmino
cd proxmino
sudo ./deploy/install-casaos.sh
```

The installer:
1. builds the app and starts it next to its own Ollama,
2. downloads the local model (phi4-mini, about 2.5 GB; skip-able, see below),
3. turns on local models (and cloud models once a key is added),
4. asks for the **Proxmino password**: this is what you type on your phone to sign in (10+ characters).

When it says **Done**, open `http://192.168.1.50:3000` on your phone or Mac, sign in, and:
- Settings > **API keys**: paste your DeepSeek key (and email, Discord, Reddit, X if you use them), press
  **Save keys**, then **Test connections**.
- Proxmino also appears as an app tile in the CasaOS dashboard (start, stop and logs are there).

Other model choices for the installer (pick one):
```bash
sudo ./deploy/install-casaos.sh phi4-mini        # default, good on 8 GB
sudo ./deploy/install-casaos.sh granite3.3:8b    # smarter, needs 16 GB, slower
sudo ./deploy/install-casaos.sh models/proxmino-8b-q4_k_m.gguf   # your own trained model (copy it into models/ first)
```

From the Mac in one step instead (if you set up an SSH key, `ssh-copy-id you@192.168.1.50`):
`./deploy/send-to-server.sh you@192.168.1.50`

## Part 3: use it away from home (optional, free)

Do not open a port on your router. Use Tailscale instead:
1. Make a free account at tailscale.com and install the app on your phone.
2. On the ThinkPad: `curl -fsSL https://tailscale.com/install.sh | sh` then `sudo tailscale up` and open the
   link it prints to sign in.
3. `sudo tailscale serve --bg 3000` gives you an https address (shown by `tailscale serve status`).
4. Add that name to the app: `cd ~/proxmino && sudo PROXMINO_HOSTS=thinkpad.your-tailnet.ts.net,192.168.1.50 ./deploy/install-casaos.sh`

## Everyday

| To | Do |
|---|---|
| Update Proxmino | copy the new zip over, unzip into `proxmino` again, `sudo ./deploy/install-casaos.sh` (chats, bots and keys are kept) |
| Change the sign-in password | `cd ~/proxmino && sudo docker compose -f deploy/docker-compose.yml exec proxmino python -m proxmino password` |
| See logs | CasaOS dashboard > Proxmino > Logs |
| Restart | CasaOS dashboard > Proxmino > Restart |
| Back up | the Docker volumes `proxmino_data` and `proxmino_bots` hold everything |
| Uninstall | `cd ~/proxmino && sudo docker compose -f deploy/docker-compose.yml down -v --rmi all` (deletes chats too) |

## If something goes wrong

- **The page does not open:** is the ThinkPad on and on the same Wi-Fi? Try `http://<address>:3000` with
  the address from `hostname -I` on the ThinkPad.
- **"Host not allowed":** you used a name the app does not know. Run the installer again with
  `PROXMINO_HOSTS=<that name>,<address>` in front, as in Part 3.
- **Sign-in keeps asking again:** clear the site's cookies once; use the same address every time.
- **Answers are a "demo" or say no model is on:** add a key in Settings > API keys, or wait for the model
  download to finish (CasaOS > Proxmino > Logs, or `sudo docker compose -f deploy/docker-compose.yml logs ollama`).
- **Very slow answers:** the model is too big for the RAM. Use phi4-mini, or cloud models.
