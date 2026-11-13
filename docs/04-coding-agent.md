# 4. The coding agent

Run it from inside a project folder. It reads the code, edits it, runs commands and tests, and stops for your
approval before anything risky.

```bash
cd ~/my-project
~/proxmino/run.sh code "the date parser fails on ISO week dates; fix it"
```

Plan first, then decide:

```bash
~/proxmino/run.sh code --plan "add a --json flag to the export command"
```

## Modes

| `--mode` | What needs your OK |
|---|---|
| `ask` (default) | Every edit and every command |
| `auto-edit` | Commands other than running the tests |
| `auto` | Only commands outside the allowlist (tests, linters, read-only git run freely) |

## Safety, enforced in code (not just asked of the model)

- Paths can't leave the project folder.
- On macOS, commands run in a sandbox with no network and can only write inside the project.
- A denylist always blocks things like `sudo`, `rm -rf /` and `curl | sh`.
- Deleting, moving, installing and pushing always ask first.

## How edits work

The model replies with search/replace blocks. Each block must match the file exactly, then with relaxed
indentation, then with relaxed spacing; an unclear match is refused and the closest region is shown. Python or
JSON that would no longer parse after an edit is refused. If code changed and the model tries to finish
without testing, the tests run anyway, and failures go back to the model up to two times.

Settings live in the `code:` section of `config.yaml`.
