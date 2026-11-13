"""Command line: `python -m proxmino serve`, `python -m proxmino ask "question"` or `python -m proxmino code "goal"`."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import Config


def serve(args) -> None:
    import uvicorn
    cfg = Config()
    host = args.host or cfg.get("host", "127.0.0.1")
    port = args.port or cfg.get("port", 3000)
    print(f"{cfg.product_name} running at http://{host}:{port}")
    uvicorn.run("proxmino.server:app", host=host, port=port, log_level="warning")


async def ask(args) -> None:
    from .pipeline import Engine, TurnRequest
    engine = Engine.create(Config())
    req = TurnRequest(message=" ".join(args.question), search=args.search, deep=args.deep, think=args.think,
                      heavy=args.heavy, persona=args.persona, agent=args.agent)
    sources = []
    try:
        async for kind, data in engine.turn(req):
            if kind == "token":
                sys.stdout.write(data["t"])
                sys.stdout.flush()
            elif kind == "thinking" and args.show_thinking:
                sys.stderr.write(f"\033[2m{data['t']}\033[0m")
            elif kind == "step":
                sys.stderr.write(f"\033[2m[{data['agent']}] {data['text']}\033[0m\n")
            elif kind == "source":
                sources.append(data)
            elif kind == "error":
                sys.stderr.write(f"\nError: {data['message']}\n")
            elif kind == "done":
                used = set(data.get("used_sources") or [])
                print()
                for s in sources:
                    if not used or s["n"] in used:
                        print(f"[{s['n']}] {s['title']} - {s['url']}")
                print(f"\033[2m{data.get('model')} · {data.get('latency_ms', 0) / 1000:.1f}s\033[0m", file=sys.stderr)
    finally:
        await engine.close()


async def code(args) -> None:
    import os

    import httpx

    from .code.agent import CodeAgent, CodeRequest
    from .code.pipeline import Pipeline, choose_strategy
    from .code.permissions import Policy
    from .code.workspace import Workspace
    from .llm import Router
    from .prompts import load_agents
    from .config import ROOT

    cfg = Config()
    dim = lambda s: f"\033[2m{s}\033[0m"
    ws = Workspace(args.dir or os.getcwd(), cfg.get("code.read_paths", []), cfg.get("code.test_command"),
                   allow_test_edits=args.allow_test_edits or bool(cfg.get("code.allow_test_edits", False)))

    async def approve(tool, a, preview, reason):
        print(f"\n--- {tool} needs approval ({reason}) ---\n{preview or a}")
        ans = (await asyncio.to_thread(input, "Allow? [y]es / [n]o / [a]lways for this command: ")).strip().lower()
        if ans.startswith("a") and tool == "run":
            policy.remembered.add(a.get("command", ""))
        return ans[:1] in ("y", "a")

    async def ask_user(q):
        return await asyncio.to_thread(input, f"\n{q}\n> ")

    forge = load_agents(ROOT / "crew").get("forge")
    async with httpx.AsyncClient(timeout=cfg.get("limits.request_timeout", 120)) as client:
        router = Router.from_config(cfg, client)

        async def loop(mode, plan=""):
            nonlocal policy
            policy = Policy.from_config(cfg, mode)
            strategy = "loop" if mode == "plan" or plan else choose_strategy(cfg, router, args.strategy)
            if strategy == "pipeline":
                agent = Pipeline(router, ws, policy, cfg.product_name, approve=approve)
            else:
                agent = CodeAgent(router, ws, policy, cfg.product_name, forge.persona_block() if forge else "",
                                  approve=approve, ask=ask_user)
            final = None
            async for kind, d in agent.run(CodeRequest(" ".join(args.goal), mode, plan, args.max_steps)):
                if kind == "model" and d["step"] == 1:
                    print(dim(f"{d['model']} · {mode} mode · {strategy} · {ws.root}"), file=sys.stderr)
                elif kind == "call":
                    print(dim(f"> {d['tool']} {json.dumps(d['args'])[:160]}"), file=sys.stderr)
                elif kind == "result":
                    first = (d["output"].splitlines() or [""])[0]
                    print(dim(f"  {'ok' if d['ok'] else 'failed'}: {first[:160]}"), file=sys.stderr)
                elif kind == "final":
                    final = d
                    print("\n" + d["text"])
                    if d["changed"]:
                        tests = {True: "pass", False: "fail", None: "not run"}[d["tests_passed"]]
                        print(dim(f"changed: {', '.join(d['changed'])} · tests {tests} · {d['steps']} steps"), file=sys.stderr)
            return final

        policy = None
        if args.plan:
            plan = await loop("plan")
            if not plan:
                return
            ans = (await asyncio.to_thread(input, f"\nExecute this plan in {args.mode} mode? [y/N] ")).strip().lower()
            if ans.startswith("y"):
                await loop(args.mode, plan["text"])
        else:
            await loop(args.mode)


def set_password() -> None:
    import getpass
    from .auth import Auth
    from .config import DATA_DIR
    pw = getpass.getpass("New password for remote access (10+ characters): ")
    if pw != getpass.getpass("Repeat it: "):
        sys.exit("The two passwords differ; nothing changed.")
    try:
        Auth(DATA_DIR / "auth.json").set_password(pw)
    except ValueError as e:
        sys.exit(str(e))
    print("Saved. Remote devices must sign in again; this computer needs no password.")


def main() -> None:
    from .config import migrate_old_names
    migrate_old_names()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="proxmino")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    a = sub.add_parser("ask", help="ask one question in the terminal")
    a.add_argument("question", nargs="+")
    for flag in ("search", "deep", "think", "heavy"):
        a.add_argument(f"--{flag}", action="store_true")
    a.add_argument("--persona")
    a.add_argument("--agent")
    a.add_argument("--show-thinking", action="store_true")
    c = sub.add_parser("code", help="work on a code project with tools (edits, commands, tests)")
    c.add_argument("goal", nargs="+")
    c.add_argument("--dir", help="project folder (default: current folder)")
    c.add_argument("--mode", default="ask", choices=["ask", "auto-edit", "auto"],
                   help="ask: approve every edit and command; auto-edit: edits and tests free; auto: allowlisted commands free too")
    c.add_argument("--plan", action="store_true", help="explore read-only and write a plan first, then ask before executing")
    c.add_argument("--max-steps", type=int, default=24)
    c.add_argument("--strategy", choices=["auto", "loop", "pipeline"], default=None,
                   help="loop: the model picks tools; pipeline: fixed localize, patch, test steps for small models (default from config)")
    c.add_argument("--allow-test-edits", action="store_true", help="let the agent change test files (read-only by default)")
    sub.add_parser("password", help="set the password for opening Proxmino from another device (tunnel or Tailscale)")
    args = ap.parse_args()
    if args.cmd == "password":
        set_password()
    elif args.cmd == "ask":
        asyncio.run(ask(args))
    elif args.cmd == "code":
        asyncio.run(code(args))
    else:
        if not args.cmd:
            args.host = args.port = None
        serve(args)


if __name__ == "__main__":
    main()
