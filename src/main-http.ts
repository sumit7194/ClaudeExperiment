#!/usr/bin/env node
import { startHttp } from './http.js';
import { closeDb, getDb } from './db.js';

const token = process.env.BRIDGE_TOKEN ?? '';
if (token.length < 32) {
  process.stderr.write(
    'BRIDGE_TOKEN must be set to a string of at least 32 characters.\n' +
      'Generate one with: openssl rand -hex 32\n'
  );
  process.exit(1);
}

const host = process.env.BRIDGE_HOST ?? '127.0.0.1';
const port = parseInt(process.env.BRIDGE_PORT ?? '47821', 10);
const path = process.env.BRIDGE_PATH ?? '/mcp';

getDb();

const server = startHttp({ host, port, token, path });

const shutdown = () => {
  server.close(() => {
    closeDb();
    process.exit(0);
  });
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
