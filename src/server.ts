import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import {
  saveTranscript,
  listTranscripts,
  searchTranscripts,
  getTranscript,
  deleteTranscript,
  tagTranscript,
} from './store.js';
import { renderMarkdown, wrapForModel } from './render.js';

const messageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system', 'tool']),
  content: z.string(),
  contentJson: z.unknown().optional(),
  createdAt: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
});

const saveSchema = z.object({
  source: z.enum(['claude_desktop', 'claude_code', 'manual']).default('claude_desktop'),
  sourceChatId: z.string().optional(),
  title: z.string().optional(),
  model: z.string().optional(),
  messages: z.array(messageSchema).min(1),
  tags: z.array(z.string()).optional(),
});

const listSchema = z.object({
  limit: z.number().int().positive().max(100).optional(),
  source: z.enum(['claude_desktop', 'claude_code', 'manual']).optional(),
  tag: z.string().optional(),
});

const searchSchema = z.object({
  query: z.string().min(1),
  limit: z.number().int().positive().max(50).optional(),
});

const getSchema = z.object({
  id: z.string(),
  include_messages: z.boolean().optional(),
  range: z
    .object({ start: z.number().int().nonnegative(), end: z.number().int().nonnegative() })
    .optional(),
  format: z.enum(['markdown', 'json']).optional(),
});

const deleteSchema = z.object({
  id: z.string(),
  confirm: z.literal(true),
});

const tagSchema = z.object({
  id: z.string(),
  add: z.array(z.string()).optional(),
  remove: z.array(z.string()).optional(),
});

const TOOLS = [
  {
    name: 'bridge_save_current_conversation',
    description:
      'Save the current conversation to the bridge store. The caller (model) supplies the messages array verbatim from its context. Returns the new transcript id.',
    inputSchema: {
      type: 'object',
      required: ['messages'],
      properties: {
        source: {
          type: 'string',
          enum: ['claude_desktop', 'claude_code', 'manual'],
          description: 'Origin surface. Default: claude_desktop.',
        },
        sourceChatId: {
          type: 'string',
          description: 'Stable id for this chat; re-saving with the same id upserts.',
        },
        title: { type: 'string' },
        model: { type: 'string' },
        messages: {
          type: 'array',
          items: {
            type: 'object',
            required: ['role', 'content'],
            properties: {
              role: { type: 'string', enum: ['user', 'assistant', 'system', 'tool'] },
              content: { type: 'string' },
              contentJson: {},
              createdAt: { type: 'string' },
              metadata: { type: 'object' },
            },
          },
        },
        tags: { type: 'array', items: { type: 'string' } },
      },
    },
  },
  {
    name: 'bridge_list_transcripts',
    description: 'List saved transcripts, most recent first.',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'number' },
        source: { type: 'string', enum: ['claude_desktop', 'claude_code', 'manual'] },
        tag: { type: 'string' },
      },
    },
  },
  {
    name: 'bridge_search_transcripts',
    description: 'Full-text search across saved transcripts. Returns matches with snippets.',
    inputSchema: {
      type: 'object',
      required: ['query'],
      properties: {
        query: { type: 'string' },
        limit: { type: 'number' },
      },
    },
  },
  {
    name: 'bridge_get_transcript',
    description:
      'Get a saved transcript. Default returns metadata only; pass include_messages=true for full content. Use range for slices of large transcripts.',
    inputSchema: {
      type: 'object',
      required: ['id'],
      properties: {
        id: { type: 'string' },
        include_messages: { type: 'boolean' },
        range: {
          type: 'object',
          required: ['start', 'end'],
          properties: { start: { type: 'number' }, end: { type: 'number' } },
        },
        format: { type: 'string', enum: ['markdown', 'json'] },
      },
    },
  },
  {
    name: 'bridge_tag_transcript',
    description: 'Add or remove tags on a transcript.',
    inputSchema: {
      type: 'object',
      required: ['id'],
      properties: {
        id: { type: 'string' },
        add: { type: 'array', items: { type: 'string' } },
        remove: { type: 'array', items: { type: 'string' } },
      },
    },
  },
  {
    name: 'bridge_delete_transcript',
    description: 'Delete a transcript by id. Irreversible. Requires confirm=true.',
    inputSchema: {
      type: 'object',
      required: ['id', 'confirm'],
      properties: {
        id: { type: 'string' },
        confirm: { type: 'boolean', enum: [true] },
      },
    },
  },
];

function ok(text: string) {
  return { content: [{ type: 'text', text }] };
}

function err(code: string, message: string) {
  return {
    content: [{ type: 'text', text: JSON.stringify({ code, message }) }],
    isError: true,
  };
}

export function createServer(): Server {
  const server = new Server(
    { name: 'claude-bridge', version: '0.1.0' },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const { name, arguments: args } = req.params;
    try {
      switch (name) {
        case 'bridge_save_current_conversation': {
          const input = saveSchema.parse(args ?? {});
          const r = saveTranscript(input);
          return ok(JSON.stringify(r));
        }
        case 'bridge_list_transcripts': {
          const input = listSchema.parse(args ?? {});
          return ok(JSON.stringify(listTranscripts(input)));
        }
        case 'bridge_search_transcripts': {
          const input = searchSchema.parse(args ?? {});
          return ok(JSON.stringify(searchTranscripts(input.query, input.limit)));
        }
        case 'bridge_get_transcript': {
          const input = getSchema.parse(args ?? {});
          const result = getTranscript(input.id, {
            includeMessages: input.include_messages,
            range: input.range,
          });
          if (!result) return err('NOT_FOUND', `transcript ${input.id} not found`);
          if (!input.include_messages) {
            return ok(JSON.stringify(result.transcript));
          }
          if ((input.format ?? 'markdown') === 'json') {
            return ok(JSON.stringify(result));
          }
          const md = renderMarkdown(result.transcript, result.messages ?? []);
          return ok(wrapForModel(result.transcript.id, result.transcript.source, md));
        }
        case 'bridge_tag_transcript': {
          const input = tagSchema.parse(args ?? {});
          const tags = tagTranscript(input.id, input.add, input.remove);
          return ok(JSON.stringify({ tags }));
        }
        case 'bridge_delete_transcript': {
          const input = deleteSchema.parse(args ?? {});
          const deleted = deleteTranscript(input.id);
          if (!deleted) return err('NOT_FOUND', `transcript ${input.id} not found`);
          return ok(JSON.stringify({ deleted: true }));
        }
        default:
          return err('UNKNOWN_TOOL', `unknown tool: ${name}`);
      }
    } catch (e) {
      if (e instanceof z.ZodError) {
        return err('INVALID_INPUT', e.issues.map((i) => i.message).join('; '));
      }
      const msg = e instanceof Error ? e.message : 'unknown error';
      return err('DB_ERROR', msg);
    }
  });

  return server;
}

export async function runStdio(): Promise<void> {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
