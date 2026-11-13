# Rakazo (Apache License 2.0)

Parts of Proxmino are adapted from Rakazo, https://github.com/elie222/rakazo
(commit ce29b8d, 2026-10-29), Copyright the Rakazo authors, licensed under the
Apache License, Version 2.0 (full text in LICENSE next to this file).

| Proxmino file | Adapted from | Changes |
|---|---|---|
| proxmino/routines.py | packages/core/src/cron.ts | Translated from TypeScript to Python; English labels only; own 5-field cron evaluator instead of the croner library |
| proxmino/scheduler.py, routine tables in proxmino/db.py, the Routines screen in web/index.html | the routine model (packages/core/src/routine-runs.ts, docs) | Reimplemented for Proxmino: one chat per routine, runs skipped when no model is on, one catch-up run after downtime |
| proxmino/approval.py | packages/core/src/action-approval.ts | Translated to Python; name patterns, declared-hint handling and allow/ask/deny rules for tools and servers only |
| proxmino/mcp.py | packages/adapters/src/mcp-transport.ts (safety rules) | Own JSON-RPC client instead of the MCP SDK; kept the launch-command allowlist, HTTPS-only remote URLs, timeouts and stderr tail |
| Scratchpad (proxmino/db.py, built-in tools in proxmino/pipeline.py, Memory pane) | packages/adapters/src/scratchpad-tools.ts, run-tools-tasks.ts | Same statuses, limits and tool names; stored per agent in SQLite; no approval for built-in tools, as in Rakazo |
| Bot avatars in web/index.html: avatarIdentitySeed(), organicAvatarPath() and the .px-organic-avatar styles and keyframes | packages/core/src/avatar-shape.ts, packages/ui-web/src/styles.css (organic avatar rules), markup from packages/ui-web/src/bot-avatar.tsx | Converted from TypeScript/React to plain JavaScript that writes SVG; class prefix rakazo- renamed px-; verified to produce identical paths. Rakazo's mascot (logo-avatar-*) and its name and logo are not used: Apache-2.0 grants no trademark rights |
| Artifact preview policy in web/index.html (ARTIFACT_CSP, withSandboxedCsp, iframe sandbox="allow-scripts") | packages/core/src/artifact-sandbox.ts | Same CSP and sandbox rule, in plain JavaScript |
| speakable() and toUtterances() in web/index.html | packages/core/src/speech-text.ts | Ported to plain JavaScript; also drops citation markers; uses the browser's own voices instead of ElevenLabs or OpenAI |

Rakazo itself is not bundled; only the parts listed above.
