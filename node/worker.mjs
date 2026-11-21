// node.hexfield.org: one HSDM node on Cloudflare Workers. Read-only, public, no state kept.
//   GET /api/status            what this node is and holds (hsdm/web.py's shape, plus more)
//   GET /api/search?q=&k=      Node.search's answer, and where it sits on the grid
//   GET /api/field             the grid: what each of the 217 cells holds

import wasm from './core.wasm';
import vectors from './memory/memory.bin';
import meta from './memory/memory.json';             // bundled as an object
import { HsdmNode } from './node.mjs';

const node = new HsdmNode({ wasm, vectors, meta }).warm();

const BASE = {
  'Access-Control-Allow-Origin': '*',
  'X-Content-Type-Options': 'nosniff',
  'Content-Type': 'application/json; charset=utf-8',
};

function reply(body, status = 200, maxAge = 0) {
  const headers = { ...BASE, 'Cache-Control': maxAge ? `public, max-age=${maxAge}` : 'no-store' };
  return new Response(JSON.stringify(body), { status, headers });
}

export default {
  fetch(request) {
    const url = new URL(request.url);
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: { ...BASE, 'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS', 'Access-Control-Max-Age': '86400' } });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') return reply({ error: 'GET only' }, 405);
    try {
      switch (url.pathname) {
        case '/':
          return reply({ node: node.name, api: ['/api/status', '/api/search?q=...&k=5', '/api/field'], site: 'https://hexfield.org' }, 200, 300);
        case '/api/status':
          return reply(node.status());
        case '/api/field':
          return reply(node.field(), 200, 300);
        case '/api/search': {
          const q = (url.searchParams.get('q') || '').trim();
          if (!q) return reply({ error: 'ask something: /api/search?q=...' }, 400);
          const k = Number(url.searchParams.get('k') || 5);
          if (!Number.isInteger(k)) return reply({ error: 'k must be a whole number' }, 400);
          return reply(node.search(q.slice(0, 2000), Math.max(1, Math.min(20, k))));
        }
        default:
          return reply({ error: 'not found' }, 404);
      }
    } catch (err) {
      return reply({ error: 'the node failed to answer', detail: String(err && err.message || err) }, 500);
    }
  },
};
