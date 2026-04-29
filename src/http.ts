import http from 'node:http';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { createServer as createMcpServer } from './server.js';

async function readJsonBody(req: http.IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? JSON.parse(raw) : undefined;
}

function unauthorized(res: http.ServerResponse) {
  res.writeHead(401, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: 'unauthorized' }));
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}

export interface HttpOptions {
  host: string;
  port: number;
  token: string;
  path: string;
}

export function startHttp(opts: HttpOptions): http.Server {
  const expected = `Bearer ${opts.token}`;

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');

      if (url.pathname === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
      }

      if (url.pathname !== opts.path) {
        res.writeHead(404).end();
        return;
      }

      const auth = req.headers.authorization ?? '';
      if (!timingSafeEqual(auth, expected)) {
        unauthorized(res);
        return;
      }

      if (req.method === 'POST') {
        const body = await readJsonBody(req);
        const mcp = createMcpServer();
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: undefined,
        });
        res.on('close', () => {
          transport.close().catch(() => {});
          mcp.close().catch(() => {});
        });
        await mcp.connect(transport);
        await transport.handleRequest(req, res, body);
        return;
      }

      res.writeHead(405, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'method not allowed' }));
    } catch (e) {
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' });
      }
      res.end(
        JSON.stringify({ error: e instanceof Error ? e.message : 'internal error' })
      );
    }
  });

  server.listen(opts.port, opts.host, () => {
    process.stderr.write(
      `claude-bridge http: listening on http://${opts.host}:${opts.port}${opts.path}\n`
    );
  });

  return server;
}
