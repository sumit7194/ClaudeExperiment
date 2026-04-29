import type Database from 'better-sqlite3';

const SCHEMA_V1 = `
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
  id              TEXT PRIMARY KEY,
  title           TEXT NOT NULL,
  source          TEXT NOT NULL,
  source_chat_id  TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  captured_at     TEXT NOT NULL,
  model           TEXT,
  message_count   INTEGER NOT NULL,
  token_estimate  INTEGER,
  summary         TEXT,
  raw_blob        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcripts_captured_at
  ON transcripts(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcripts_source_chat_id
  ON transcripts(source_chat_id);

CREATE TABLE IF NOT EXISTS messages (
  id            TEXT PRIMARY KEY,
  transcript_id TEXT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  seq           INTEGER NOT NULL,
  role          TEXT NOT NULL,
  content       TEXT NOT NULL,
  content_json  TEXT,
  created_at    TEXT,
  metadata      TEXT,
  UNIQUE(transcript_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_transcript
  ON messages(transcript_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content,
  transcript_id UNINDEXED,
  seq UNINDEXED,
  role UNINDEXED,
  content='messages',
  content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, transcript_id, seq, role)
  VALUES (new.rowid, new.content, new.transcript_id, new.seq, new.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.rowid;
  INSERT INTO messages_fts(rowid, content, transcript_id, seq, role)
  VALUES (new.rowid, new.content, new.transcript_id, new.seq, new.role);
END;

CREATE TABLE IF NOT EXISTS tags (
  id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS transcript_tags (
  transcript_id TEXT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  tag_id        INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY (transcript_id, tag_id)
);
`;

export function migrate(db: Database.Database): void {
  db.exec(SCHEMA_V1);
  const row = db
    .prepare('SELECT MAX(version) AS v FROM schema_version')
    .get() as { v: number | null };
  if (row.v == null) {
    db.prepare(
      'INSERT INTO schema_version (version, applied_at) VALUES (?, ?)'
    ).run(1, new Date().toISOString());
  }
}
