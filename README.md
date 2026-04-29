# Claude Bridge

A local MCP server that shares full conversation transcripts between Claude Desktop and Claude Code on macOS.

Both surfaces speak to the same SQLite-backed store. Saves are triggered by asking Claude in the desktop app to save the current chat; reads happen from any session via the `bridge_*` tools.

## Status

v0.1 scaffold. Not yet wired up. See `claudebridgeplan.md` for the full original spec; the implementation here is scoped to the simplified MCP-only design (no daemon, no browser extension, no CLI).

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

## Privacy

Everything is local. The SQLite file lives in `~/.claude-bridge/`. Delete that directory to wipe state.

Common secret formats (AWS, GitHub PAT, OpenAI/Anthropic keys, private key blocks) are redacted on save. See `src/redact.ts` for the patterns.
