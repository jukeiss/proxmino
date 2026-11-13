"""
Desktop launcher used by the installers (.dmg, .exe, .deb, .rpm, AppImage, Flatpak).

Starts Proxmino on a free local port, opens it in your browser and shows a small
window with "Open in browser" and "Quit" so it never runs invisibly.
Proxmino still needs Ollama (https://ollama.com) running with a model.
"""
import os
import sys
import threading
import time
import urllib.request
import webbrowser

# Windows consoles default to a legacy code page; printing characters like "→" would crash the server.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HOME_DIR = os.path.join(os.path.expanduser("~"), ".proxmino")
if getattr(sys, "frozen", False):
    # Installed apps can't write inside their own folder: keep data next to your bots in ~/.proxmino.
    os.environ.setdefault("PROXMINO_DATA", os.path.join(HOME_DIR, "data"))
    os.environ.setdefault("PROXMINO_ENV_FILE", os.path.join(HOME_DIR, ".env"))

ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def free_port(start):
    import socket
    for p in range(start, start + 50):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("ask", "code", "password", "serve"):
        from proxmino.__main__ import main as cli      # `Proxmino ask "..."` etc. still work
        return cli()
    try:
        import proxmino.server  # noqa: F401  load it here so the bundle includes it and errors are visible
    except Exception:
        import traceback
        traceback.print_exc()
        if os.environ.get("APP_SMOKE"):
            os._exit(1)
        raise
    port = free_port(3000)
    url = f"http://127.0.0.1:{port}/"
    from proxmino.__main__ import main as cli
    sys.argv = ["proxmino", "serve", "--host", "127.0.0.1", "--port", str(port)]
    threading.Thread(target=cli, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=0.3)
            break
        except Exception:
            time.sleep(0.15)
    if os.environ.get("APP_SMOKE"):          # CI: prove the bundled app starts and serves its page, then exit
        try:
            ok = urllib.request.urlopen(url, timeout=10).status == 200
        except Exception as e:
            print("smoke test failed:", e, flush=True)
            ok = False
        print("smoke test:", "OK" if ok else "FAILED", url, flush=True)
        os._exit(0 if ok else 1)
    webbrowser.open(url)
    try:
        control_window(url)
    except Exception:
        print(f"Proxmino is running at {url}  (close this window or press Ctrl+C to stop)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def control_window(url):
    import tkinter as tk
    root = tk.Tk()
    root.title("Proxmino")
    root.resizable(False, False)
    bg, fg = "#18181b", "#fafafa"
    root.configure(bg=bg, padx=22, pady=18)
    try:
        icon = tk.PhotoImage(file=os.path.join(ROOT, "packaging", "icon.png")).subsample(16, 16)
        root.iconphoto(True, icon)
        tk.Label(root, image=icon, bg=bg).grid(row=0, column=0, rowspan=3, padx=(0, 14))
    except Exception:
        pass
    tk.Label(root, text="Proxmino is running", font=("Helvetica", 15, "bold"), bg=bg, fg=fg).grid(row=0, column=1, sticky="w")
    tk.Label(root, text=url, font=("Menlo", 11), bg=bg, fg="#a1a1aa").grid(row=1, column=1, sticky="w")
    tk.Label(root, text="Needs Ollama running with a model (ollama.com)", font=("Helvetica", 11), bg=bg,
             fg="#71717a").grid(row=2, column=1, sticky="w")
    row = tk.Frame(root, bg=bg)
    row.grid(row=3, column=0, columnspan=2, pady=(14, 0), sticky="e")
    tk.Button(row, text="Open in browser", command=lambda: webbrowser.open(url), highlightbackground=bg,
              relief="flat", padx=10).pack(side="left", padx=(0, 8))
    tk.Button(row, text="Quit", command=lambda: os._exit(0), highlightbackground=bg, relief="flat", padx=10).pack(side="left")
    root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0))
    root.mainloop()
    os._exit(0)


if __name__ == "__main__":
    main()
