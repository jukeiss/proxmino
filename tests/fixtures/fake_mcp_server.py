"""A tiny MCP server over stdio for tests: two tools, notes kept in memory."""
import json
import sys

NOTES = []
TOOLS = [
    {"name": "list_notes", "description": "List saved notes", "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"readOnlyHint": True}},
    {"name": "add_note", "description": "Save a note", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                                                                      "required": ["text"]}},
]
print("starting fake server (log line on stdout)", flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    m, p = msg["method"], msg.get("params") or {}
    if m == "initialize":
        res = {"protocolVersion": p["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "notes"}}
    elif m == "tools/list":
        res = {"tools": TOOLS}
    elif m == "tools/call" and p["name"] == "add_note":
        NOTES.append(p["arguments"]["text"])
        res = {"content": [{"type": "text", "text": f"saved ({len(NOTES)} notes)"}]}
    elif m == "tools/call" and p["name"] == "list_notes":
        res = {"content": [{"type": "text", "text": "\n".join(NOTES) or "no notes"}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"unknown {m}"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}), flush=True)
