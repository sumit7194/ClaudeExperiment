import { ulid } from 'ulid';
import { getDb } from './db.js';
import { redact } from './redact.js';
import type { SaveTranscriptInput, Transcript, Message } from './types.js';

function deriveTitle(messages: SaveTranscriptInput['messages']): string {
  const firstUser = messages.find((m) => m.role === 'user');
  const text = (firstUser?.content ?? '').trim();
  if (!text) return `Conversation from ${new Date().toISOString().slice(0, 10)}`;
  const firstLine = text.split('\n')[0].replace(/`+/g, '').trim();
  return firstLine.length > 80 ? firstLine.slice(0, 77) + '...' : firstLine;
}

function tagIds(names: string[]): number[] {
  if (names.length === 0) return [];
  const db = getDb();
  const insert = db.prepare('INSERT OR IGNORE INTO tags (name) VALUES (?)');
  const select = db.prepare('SELECT id FROM tags WHERE name = ?');
  return names.map((name) => {
    insert.run(name);
    return (select.get(name) as { id: number }).id;
  });
}

export interface SaveResult {
  id: string;
  title: string;
  redactionCount: number;
  upserted: boolean;
}

export function saveTranscript(input: SaveTranscriptInput): SaveResult {
  const db = getDb();
  const now = new Date().toISOString();
  const title = input.title?.trim() || deriveTitle(input.messages);

  let totalRedactions = 0;
  const cleanedMessages = input.messages.map((m) => {
    const r = redact(m.content);
    totalRedactions += r.count;
    return { ...m, content: r.content };
  });

  const totalChars = cleanedMessages.reduce((n, m) => n + m.content.length, 0);
  const tokenEstimate = Math.ceil(totalChars / 4);

  const existing = input.sourceChatId
    ? (db
        .prepare('SELECT id, captured_at FROM transcripts WHERE source_chat_id = ?')
        .get(input.sourceChatId) as { id: string; captured_at: string } | undefined)
    : undefined;

  const id = existing?.id ?? ulid();
  const capturedAt = existing?.captured_at ?? now;
  const rawBlob = JSON.stringify(input);

  const tx = db.transaction(() => {
    if (existing) {
      db.prepare('DELETE FROM messages WHERE transcript_id = ?').run(id);
      db.prepare(
        `UPDATE transcripts SET title = ?, source = ?, updated_at = ?, model = ?,
           message_count = ?, token_estimate = ?, raw_blob = ? WHERE id = ?`
      ).run(
        title,
        input.source,
        now,
        input.model ?? null,
        cleanedMessages.length,
        tokenEstimate,
        rawBlob,
        id
      );
    } else {
      db.prepare(
        `INSERT INTO transcripts
         (id, title, source, source_chat_id, created_at, updated_at, captured_at,
          model, message_count, token_estimate, raw_blob)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        id,
        title,
        input.source,
        input.sourceChatId ?? null,
        now,
        now,
        capturedAt,
        input.model ?? null,
        cleanedMessages.length,
        tokenEstimate,
        rawBlob
      );
    }

    const insertMsg = db.prepare(
      `INSERT INTO messages
       (id, transcript_id, seq, role, content, content_json, created_at, metadata)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    );
    cleanedMessages.forEach((m, seq) => {
      insertMsg.run(
        ulid(),
        id,
        seq,
        m.role,
        m.content,
        m.contentJson ? JSON.stringify(m.contentJson) : null,
        m.createdAt ?? null,
        m.metadata ? JSON.stringify(m.metadata) : null
      );
    });

    if (input.tags?.length) {
      const link = db.prepare(
        'INSERT OR IGNORE INTO transcript_tags (transcript_id, tag_id) VALUES (?, ?)'
      );
      for (const tagId of tagIds(input.tags)) link.run(id, tagId);
    }
  });
  tx();

  return { id, title, redactionCount: totalRedactions, upserted: !!existing };
}

export interface ListOpts {
  limit?: number;
  source?: string;
  tag?: string;
}

export function listTranscripts(opts: ListOpts = {}): Array<
  Pick<Transcript, 'id' | 'title' | 'source' | 'capturedAt' | 'messageCount' | 'tags'>
> {
  const db = getDb();
  const limit = Math.min(Math.max(opts.limit ?? 20, 1), 100);
  const params: unknown[] = [];
  let where = '1=1';
  if (opts.source) {
    where += ' AND t.source = ?';
    params.push(opts.source);
  }
  if (opts.tag) {
    where +=
      ' AND t.id IN (SELECT tt.transcript_id FROM transcript_tags tt JOIN tags g ON g.id = tt.tag_id WHERE g.name = ?)';
    params.push(opts.tag);
  }
  const rows = db
    .prepare(
      `SELECT t.id, t.title, t.source, t.captured_at, t.message_count
       FROM transcripts t
       WHERE ${where}
       ORDER BY t.captured_at DESC
       LIMIT ?`
    )
    .all(...params, limit) as Array<{
    id: string;
    title: string;
    source: string;
    captured_at: string;
    message_count: number;
  }>;
  const tagStmt = db.prepare(
    `SELECT g.name FROM tags g
     JOIN transcript_tags tt ON tt.tag_id = g.id
     WHERE tt.transcript_id = ?`
  );
  return rows.map((r) => ({
    id: r.id,
    title: r.title,
    source: r.source as Transcript['source'],
    capturedAt: r.captured_at,
    messageCount: r.message_count,
    tags: (tagStmt.all(r.id) as Array<{ name: string }>).map((t) => t.name),
  }));
}

export interface SearchHit {
  transcriptId: string;
  title: string;
  snippet: string;
  matchedSeq: number;
  role: string;
}

export function searchTranscripts(query: string, limit = 10): SearchHit[] {
  const db = getDb();
  const capped = Math.min(Math.max(limit, 1), 50);
  const rows = db
    .prepare(
      `SELECT m.transcript_id, t.title, m.seq, m.role,
              snippet(messages_fts, 0, '«', '»', '…', 12) AS snippet
       FROM messages_fts
       JOIN messages m ON m.rowid = messages_fts.rowid
       JOIN transcripts t ON t.id = m.transcript_id
       WHERE messages_fts MATCH ?
       ORDER BY rank
       LIMIT ?`
    )
    .all(query, capped) as Array<{
    transcript_id: string;
    title: string;
    seq: number;
    role: string;
    snippet: string;
  }>;
  return rows.map((r) => ({
    transcriptId: r.transcript_id,
    title: r.title,
    snippet: r.snippet,
    matchedSeq: r.seq,
    role: r.role,
  }));
}

export interface GetOpts {
  includeMessages?: boolean;
  range?: { start: number; end: number };
}

export interface GetResult {
  transcript: Transcript;
  messages?: Message[];
}

export function getTranscript(id: string, opts: GetOpts = {}): GetResult | null {
  const db = getDb();
  const row = db
    .prepare(`SELECT * FROM transcripts WHERE id = ?`)
    .get(id) as
    | {
        id: string;
        title: string;
        source: string;
        source_chat_id: string | null;
        created_at: string;
        updated_at: string;
        captured_at: string;
        model: string | null;
        message_count: number;
        token_estimate: number | null;
        summary: string | null;
      }
    | undefined;
  if (!row) return null;
  const tags = (
    db
      .prepare(
        `SELECT g.name FROM tags g
         JOIN transcript_tags tt ON tt.tag_id = g.id
         WHERE tt.transcript_id = ?`
      )
      .all(id) as Array<{ name: string }>
  ).map((t) => t.name);

  const transcript: Transcript = {
    id: row.id,
    title: row.title,
    source: row.source as Transcript['source'],
    sourceChatId: row.source_chat_id ?? undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    capturedAt: row.captured_at,
    model: row.model ?? undefined,
    messageCount: row.message_count,
    tokenEstimate: row.token_estimate ?? undefined,
    summary: row.summary ?? undefined,
    tags,
  };

  if (!opts.includeMessages) return { transcript };

  const params: unknown[] = [id];
  let rangeClause = '';
  if (opts.range) {
    rangeClause = ' AND seq BETWEEN ? AND ?';
    params.push(opts.range.start, opts.range.end);
  }
  const msgRows = db
    .prepare(
      `SELECT id, transcript_id, seq, role, content, content_json, created_at, metadata
       FROM messages WHERE transcript_id = ?${rangeClause}
       ORDER BY seq ASC`
    )
    .all(...params) as Array<{
    id: string;
    transcript_id: string;
    seq: number;
    role: string;
    content: string;
    content_json: string | null;
    created_at: string | null;
    metadata: string | null;
  }>;
  const messages: Message[] = msgRows.map((m) => ({
    id: m.id,
    transcriptId: m.transcript_id,
    seq: m.seq,
    role: m.role as Message['role'],
    content: m.content,
    contentJson: m.content_json ? JSON.parse(m.content_json) : undefined,
    createdAt: m.created_at ?? undefined,
    metadata: m.metadata ? JSON.parse(m.metadata) : undefined,
  }));
  return { transcript, messages };
}

export function deleteTranscript(id: string): boolean {
  const db = getDb();
  const r = db.prepare('DELETE FROM transcripts WHERE id = ?').run(id);
  return r.changes > 0;
}

export function tagTranscript(
  id: string,
  add: string[] = [],
  remove: string[] = []
): string[] {
  const db = getDb();
  if (add.length) {
    const link = db.prepare(
      'INSERT OR IGNORE INTO transcript_tags (transcript_id, tag_id) VALUES (?, ?)'
    );
    for (const tagId of tagIds(add)) link.run(id, tagId);
  }
  if (remove.length) {
    const stmt = db.prepare(
      `DELETE FROM transcript_tags WHERE transcript_id = ? AND tag_id IN
       (SELECT id FROM tags WHERE name = ?)`
    );
    for (const name of remove) stmt.run(id, name);
  }
  return (
    db
      .prepare(
        `SELECT g.name FROM tags g
         JOIN transcript_tags tt ON tt.tag_id = g.id
         WHERE tt.transcript_id = ?
         ORDER BY g.name`
      )
      .all(id) as Array<{ name: string }>
  ).map((t) => t.name);
}
