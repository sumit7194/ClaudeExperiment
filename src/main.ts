#!/usr/bin/env node
import { runStdio } from './server.js';
import { closeDb } from './db.js';

const shutdown = () => {
  closeDb();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

runStdio().catch((e) => {
  process.stderr.write(`claude-bridge: ${e instanceof Error ? e.stack : String(e)}\n`);
  process.exit(1);
});
