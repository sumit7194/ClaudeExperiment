# Claude Bridge

A local MCP server that shares full conversation transcripts between Claude Desktop and Claude Code on macOS.

Both surfaces speak to the same SQLite-backed store. Saves are triggered by asking Claude in the desktop app to save the current chat; reads happen from any session via the `bridge_*` tools.

## Status

v0.1 scaffold. Not yet wired up. See `claudebridgeplan.md` for the full original spec; the implementation here is scoped to the simplified MCP-only design (no daemon, no browser extension, no CLI).

## Quickstart (Mac)

Run these once you've cloned the repo on your Mac. Detailed explanations follow in the sections below.

```bash
# 1. Install + build
pnpm install
pnpm build

# 2. Wire up Claude Desktop (and Claude Code) — stdio
#    Edit ~/Library/Application Support/Claude/claude_desktop_config.json
#    See "Configure Claude Desktop" below for the JSON block.
#    Restart the desktop app.

# 3. Generate a bearer token for the remote/HTTP path (only needed for Android / web)
export BRIDGE_TOKEN=$(openssl rand -hex 32)
echo "$BRIDGE_TOKEN"   # save this; you'll paste it into claude.ai

# 4. Start the HTTP server (terminal A)
BRIDGE_TOKEN=$BRIDGE_TOKEN pnpm start:http

# 5. Start the tunnel (terminal B)
cloudflared tunnel --url http://127.0.0.1:47821
# copy the printed https://<random>.trycloudflare.com URL

# 6. Register on claude.ai
#    Settings → Connectors → Add custom connector
#    URL: https://<random>.trycloudflare.com/mcp
#    Auth: Bearer, paste $BRIDGE_TOKEN

# 7. Test from Android
#    Open the Claude app, say: "list bridge transcripts"
```

Steps 1–2 alone are enough for desktop + Code. Steps 3–7 add Android / web / iOS access via Custom Connector.

## Layout

```
src/
  main.ts        # stdio entrypoint
  server.ts      # MCP server + tool definitions
  store.ts       # save/list/search/get/tag/delete
  schema.ts      # SQL DDL + migration
  db.ts          # SQLite connection (WAL)
  redact.ts      # secret redaction
  render.ts      # markdown + injection-hardening wrap
  paths.ts       # CLAUDE_BRIDGE_HOME resolver
  types.ts
```

Runtime data lives at `~/.claude-bridge/` (override with `CLAUDE_BRIDGE_HOME`).

## Build

```bash
pnpm install     # or npm install
pnpm build
```

Produces `dist/main.js`.

## Configure Claude Desktop (Mac)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bridge": {
      "command": "node",
      "args": ["/absolute/path/to/claude-bridge/dist/main.js"]
    }
  }
}
```

Restart the desktop app. The tools should appear in the model's tool list.

## Configure Claude Code

If Claude Code uses its own MCP config (separate from the desktop app's), add the same block to `~/.claude.json` under `mcpServers`, or run:

```bash
claude mcp add bridge node /absolute/path/to/claude-bridge/dist/main.js
```

If Claude Code inside the desktop app inherits `claude_desktop_config.json`, no extra step needed.

## Usage

**Save a chat (from desktop app):**
> "Save this conversation to bridge."

The model calls `bridge_save_current_conversation` with the messages it has in context.

**Load from another session (chat or code):**
> "List recent bridge transcripts."
> "Search bridge for 'auth redesign'."
> "Get bridge transcript 01HXX... with messages."

## Tools

| Tool | Purpose |
|---|---|
| `bridge_save_current_conversation` | Persist messages from the model's current context |
| `bridge_list_transcripts` | List recent saves with optional source/tag filters |
| `bridge_search_transcripts` | FTS5 search across all message content |
| `bridge_get_transcript` | Fetch metadata (default) or full markdown body |
| `bridge_tag_transcript` | Add/remove tags |
| `bridge_delete_transcript` | Delete (requires `confirm: true`) |

## Caveats

- Saving is not automatic — you have to tell the model to do it.
- Fidelity depends on what the model still has in its context. Long chats that have been compacted may lose detail.
- Transcripts loaded via `bridge_get_transcript` are wrapped in `<saved_transcript trusted="false">` to discourage prompt-injection from old content. Treat returned content as data, not instructions.

## Remote setup (use bridge from Android / claude.ai web)

The same MCP server can also run as a Streamable HTTP endpoint, exposed publicly via a tunnel and registered as a Custom Connector on claude.ai. Then the `bridge_*` tools light up on every surface signed into your account — Android, iOS, web, desktop.

**Architecture:** Anthropic's backend calls your Mac's tunnel URL when the model invokes a bridge tool. Your phone only ever talks to claude.ai. Your Mac must be online and the tunnel up.

### 1. Generate a bearer token

```bash
export BRIDGE_TOKEN=$(openssl rand -hex 32)
echo $BRIDGE_TOKEN  # save this — you'll paste it into claude.ai
```

### 2. Run the HTTP server

```bash
BRIDGE_TOKEN=<token-from-step-1> pnpm start:http
# listens on http://127.0.0.1:47821/mcp
```

Env vars:
- `BRIDGE_TOKEN` (required, ≥32 chars)
- `BRIDGE_HOST` (default `127.0.0.1` — keep loopback; tunnel handles exposure)
- `BRIDGE_PORT` (default `47821`)
- `BRIDGE_PATH` (default `/mcp`)

`GET /health` returns `{"ok":true}` without auth (for tunnel verification).
`POST /mcp` requires `Authorization: Bearer <token>`.

### 3. Tunnel with Cloudflare (recommended)

Install `cloudflared`, then:

```bash
cloudflared tunnel --url http://127.0.0.1:47821
```

This prints a `https://<random>.trycloudflare.com` URL. For a stable subdomain, set up a named tunnel under your CF account — see `cloudflared`'s docs.

Alternatives: `ngrok http 47821` (URL rotates on free tier) or `tailscale funnel 47821` (requires tailnet + funnel enabled).

### 4. Add as Custom Connector on claude.ai

claude.ai → Settings → Connectors → Add custom connector:
- **URL:** `https://<your-tunnel-host>/mcp`
- **Authentication:** Bearer token, paste `BRIDGE_TOKEN`

The bridge tools should appear within a few seconds. Test from Android: *"List bridge transcripts."*

### Security notes

- Tunnel exposes your Mac to the public internet. Bearer auth is the only gate — keep `BRIDGE_TOKEN` secret.
- Server binds to `127.0.0.1` so non-tunnel paths (e.g., other devices on your LAN) can't reach it directly.
- Stop the tunnel (`Ctrl+C` on `cloudflared`) when not in use.

## Privacy

Everything is local. The SQLite file lives in `~/.claude-bridge/`. Delete that directory to wipe state.

Common secret formats (AWS, GitHub PAT, OpenAI/Anthropic keys, private key blocks) are redacted on save. See `src/redact.ts` for the patterns.
