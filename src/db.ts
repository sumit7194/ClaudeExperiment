import Database from 'better-sqlite3';
import { dbPath, ensureHome } from './paths.js';
import { migrate } from './schema.js';

let instance: Database.Database | null = null;

export function getDb(): Database.Database {
  if (instance) return instance;
  ensureHome();
  const db = new Database(dbPath());
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');
  migrate(db);
  instance = db;
  return db;
}

export function closeDb(): void {
  if (instance) {
    instance.close();
    instance = null;
  }
}
