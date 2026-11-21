// One HSDM node, the way the Worker runs it: the JavaScript port (hsdm.mjs) over a memory that
// build.py made with p2p-hsdm's Python, laid out on a hex grid of 217 cells.
//
// search() is Node.search on a node with no peers: the same passages and order a Python node
// holding this memory returns. Each answer also says which cell it sits in, and the walk says
// where the question landed and how far across the grid the answers were: the question is
// routed to a cell the way HSDM routes it to a peer (each cell stands in for one), then the
// walk steps, best first, into the neighbouring cell that scores highest.

import { Core, Encoder, Index, STOPWORDS, search, tokenize } from './hsdm.mjs';

export const VERSION = '1';
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];

// a binary heap of cells, best score first, then lower index (Python's heapq on (-score, cell))
class Frontier {
  constructor(score) { this.score = score; this.a = []; }
  better(i, j) { const s = this.score, x = this.a[i], y = this.a[j]; return s[x] > s[y] || (s[x] === s[y] && x < y); }
  push(c) {
    const a = this.a;
    a.push(c);
    for (let i = a.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (!this.better(i, p)) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && this.better(l, m)) m = l;
        if (r < a.length && this.better(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
  get size() { return this.a.length; }
}

export class HsdmNode {
  constructor({ wasm, vectors, meta, name = 'node.hexfield.org' }) {
    this.meta = meta;
    this.name = name;
    this.core = new Core(wasm);
    this.encoder = new Encoder(this.core, meta.key, meta.dim);
    this.encoder.idf = new Map(Object.entries(meta.idf));
    this.encoder.defaultIdf = meta.default_idf;
    this.index = new Index(this.core, vectors, meta.rows, meta.dim);
    this.cells = meta.field;
    this.cellOf = Int16Array.from(meta.cell_of);
    const at = new Map(this.cells.map((c, i) => [`${c.q},${c.r}`, i]));
    this.neighbours = this.cells.map((c) => DIRS.map(([dq, dr]) => at.get(`${c.q + dq},${c.r + dr}`)).filter((i) => i !== undefined));
    this.sizes = new Float64Array(this.cells.length);
    for (const c of this.cellOf) this.sizes[c]++;
    // each cell's sketch: in how many of its passages each word appears
    this.df = this.cells.map(() => new Map());
    this.total = new Map();
    meta.texts.forEach((text, row) => {
      const df = this.df[this.cellOf[row]];
      for (const w of new Set(tokenize(text))) { df.set(w, (df.get(w) || 0) + 1); this.total.set(w, (this.total.get(w) || 0) + 1); }
    });
    this.sizeTotal = this.sizes.reduce((s, v) => s + Math.max(v, 1), 0);
    const seed = this.core.text.encode(`${meta.key}/${name}`), at2 = this.core.alloc(seed.length), out = this.core.alloc(64);
    this.core.u8.set(seed, at2);
    this.core.ex.hash(at2, seed.length, 8, out);
    this.id = [...this.core.u8.slice(out, out + 4)].map((b) => b.toString(16).padStart(2, '0')).join('');
    this.stats = { searches: 0 };
  }

  // one search before the first visitor's, so the first real one is not also the engine's first
  warm() {
    for (const q of ['read a file, parse json and send it over a socket with a timeout', 'how do threads share a lock', 'decimal rounding'])
      this.search(q);
    this.stats.searches = 0;
    return this;
  }

  // cells apart on the grid
  distance(a, b) {
    const dq = this.cells[a].q - this.cells[b].q, dr = this.cells[a].r - this.cells[b].r;
    return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
  }

  passage(row) { return { label: this.meta.labels[this.meta.label_of[row]], text: this.meta.texts[row] }; }

  // index.route_scores with each cell as a peer (exact counts, not the 4-bit sketch); null when
  // the question has no words to route by
  route(text) {
    const words = [...new Set(tokenize(text).filter((w) => !STOPWORDS.has(w)))].sort();
    if (!words.length) return null;
    const out = new Float64Array(this.cells.length);
    for (const w of words) {
      const total = this.total.get(w) || 0;
      const background = total / this.sizeTotal, rarity = Math.log1p(this.sizeTotal / Math.max(total, 1));
      for (let c = 0; c < out.length; c++) {
        const local = (this.df[c].get(w) || 0) / Math.max(this.sizes[c], 1);
        out[c] += rarity * Math.log(0.8 * local + 0.2 * background + 1e-9);
      }
    }
    return out;
  }

  // best first from the landing cell, until the cell of the last answer
  walk(text, rows) {
    const score = this.route(text);
    if (!score) return null;
    let land = 0;
    for (let c = 1; c < score.length; c++) if (score[c] > score[land]) land = c;
    const want = new Map();
    rows.forEach((row, rank) => { const c = this.cellOf[row]; if (!want.has(c)) want.set(c, []); want.get(c).push(rank); });
    const seen = new Uint8Array(score.length), frontier = new Frontier(score), path = [], answers = [];
    let crossed = 0;
    seen[land] = 1;
    frontier.push(land);
    while (frontier.size && want.size) {
      const c = frontier.pop();
      path.push(c);
      crossed += this.sizes[c];
      for (const rank of want.get(c) || []) answers[rank] = { step: path.length - 1, away: this.distance(land, c), crossed: +(crossed / this.meta.rows).toFixed(4) };
      want.delete(c);
      for (const m of this.neighbours[c]) if (!seen[m]) { seen[m] = 1; frontier.push(m); }
    }
    const cell = (i) => ({ cell: i, q: this.cells[i].q, r: this.cells[i].r });
    return { land: cell(land), path, answers, read: +(crossed / this.meta.rows).toFixed(4) };
  }

  // GET /api/search: hsdm/web.py's answer, plus where things are on the grid
  search(text, limit = 5) {
    this.stats.searches++;
    const { results } = search(this.encoder, this.index, (row) => this.passage(row), text, limit);
    return {
      query: text,
      elapsed_ms: 0,                               // a Worker's clock stands still while it computes
      known_peers: 0,
      asked: [],
      answered: [],
      results: results.map((r) => {
        const c = this.cellOf[r.row];
        return { score: +r.score.toFixed(6), label: r.label, text: r.text, peer: 'local', cell: c, q: this.cells[c].q, r: this.cells[c].r };
      }),
      walk: this.walk(text, results.map((r) => r.row)),
    };
  }

  describe() {
    const m = this.meta;
    return {
      name: this.name,
      passages: m.rows,
      corpus: m.corpus,
      built: m.built,
      dim: m.dim,
      key: m.key,
      cells: this.cells.length,
      radius: m.radius,
      families: m.families,
      metrics: m.metrics,
      runtime: 'Cloudflare Workers',
      code: 'hsdm.mjs, a JavaScript port of p2p-hsdm 6.0.0; hexfield/node on GitHub',
      version: VERSION,
    };
  }

  // GET /api/status: hsdm/web.py's status, plus what this node is
  status() {
    return { id: this.id, addr: `https://${this.name}`, passages: this.meta.rows, sealed: false, stats: { ...this.stats }, peers: [], node: this.describe() };
  }

  // GET /api/field: the grid, one entry per cell
  field() {
    return {
      node: this.describe(),
      nodes: this.cells.map(({ id, q, r, concept, domain, modules, holds, chunk }) => ({ id, q, r, concept, domain, modules, holds, chunk })),
    };
  }
}
