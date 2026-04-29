import { homedir } from 'node:os';
import { join } from 'node:path';
import { mkdirSync } from 'node:fs';

export function bridgeHome(): string {
  return process.env.CLAUDE_BRIDGE_HOME ?? join(homedir(), '.claude-bridge');
}

export function dbPath(): string {
  return join(bridgeHome(), 'transcripts.db');
}

export function logsDir(): string {
  return join(bridgeHome(), 'logs');
}

export function ensureHome(): void {
  mkdirSync(bridgeHome(), { recursive: true, mode: 0o700 });
  mkdirSync(logsDir(), { recursive: true, mode: 0o700 });
}
