// Hexfield: the field. Memory nodes on a hex grid, drawn on a canvas.
// Hover a cell to read what it holds; click or tap it to light up its nearest neighbours.
// Keyboard: arrow keys move between cells, Enter lights one up, Escape lets go.
//
// The data comes in through loadNodes() and nowhere else. Each node is
//   { id, q, r, concept, chunk, domain, peers, recalls, stored }
// where (q, r) are its axial coordinates on the grid. To read a live HSDM or Supabase
// endpoint instead of the sample file, change NODES_URL. (Supabase's REST API also wants
// its anon key: add `apikey` to the headers in loadNodes.)

'use strict';

const NODES_URL = 'nodes.json';

async function loadNodes() {
  const res = await fetch(NODES_URL, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`${NODES_URL} answered ${res.status}`);
  const data = await res.json();
  return Array.isArray(data) ? data : data.nodes;
}

// ---------- hex arithmetic: pointy-top cells, axial coordinates

const SQRT3 = Math.sqrt(3);
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];
const key = (q, r) => `${q},${r}`;
const centre = (q, r) => [SQRT3 * (q + r / 2), 1.5 * r];          // in units of the cell's radius

function cellAt(x, y) {                                            // the nearest cell to a point (same units)
  const qf = (SQRT3 / 3) * x - y / 3, rf = (2 / 3) * y, sf = -qf - rf;
  let q = Math.round(qf), r = Math.round(rf);
  const s = Math.round(sf), dq = Math.abs(q - qf), dr = Math.abs(r - rf), ds = Math.abs(s - sf);
  if (dq > dr && dq > ds) q = -r - s;
  else if (dr > ds) r = -q - s;
  return [q, r];
}

// ---------- the field

class Field {
  constructor(nodes, { canvas, readout, legend }) {
    this.canvas = canvas;
    this.readout = readout;
    this.ctx = canvas.getContext('2d');
    this.nodes = nodes.filter((n) => Number.isInteger(n.q) && Number.isInteger(n.r));
    this.at = new Map(this.nodes.map((n) => [key(n.q, n.r), n]));
    this.hovered = null;
    this.selected = null;
    this.cursor = null;                                            // the keyboard's cell
    this.keyboard = false;                                         // is the keyboard driving (not a pointer)?
    this.lastPointer = -Infinity;
    this.touch = matchMedia('(hover: none)').matches;
    this.readColours();
    this.drawLegend(legend);

    // the grid's extent, in cell radii
    const cs = this.nodes.map((n) => centre(n.q, n.r));
    this.box = {
      x0: Math.min(...cs.map((c) => c[0])) - SQRT3 / 2, x1: Math.max(...cs.map((c) => c[0])) + SQRT3 / 2,
      y0: Math.min(...cs.map((c) => c[1])) - 1, y1: Math.max(...cs.map((c) => c[1])) + 1,
    };

    canvas.addEventListener('pointerdown', () => { this.keyboard = false; this.lastPointer = performance.now(); });
    canvas.addEventListener('pointermove', (e) => { if (e.pointerType !== 'touch') this.hover(this.nodeAtEvent(e)); });
    canvas.addEventListener('pointerleave', () => this.hover(null));
    canvas.addEventListener('click', (e) => this.select(this.nodeAtEvent(e), true));
    canvas.addEventListener('keydown', (e) => this.key(e));
    canvas.addEventListener('focus', () => {
      this.keyboard = performance.now() - this.lastPointer > 300;    // focus that a click did not bring is the keyboard's
      if (!this.cursor) this.cursor = this.selected || this.at.get(key(0, 0)) || this.nodes[0];
      this.update();
    });
    canvas.addEventListener('blur', () => { this.update(); });
    new ResizeObserver(() => this.layout()).observe(canvas.parentElement);
    this.layout();
  }

  readColours() {
    const css = getComputedStyle(document.documentElement), v = (name) => css.getPropertyValue(name).trim();
    this.colour = {
      bg: v('--bg'), cell: v('--cell'), hover: v('--cell-hover'), line: v('--line'), text: v('--text'),
      accent: v('--accent'), accentDim: v('--accent-dim'),
    };
    this.domainColour = (d) => v(`--d-${d}`) || this.colour.cell;
  }

  drawLegend(list) {
    if (!list) return;
    const domains = [...new Set(this.nodes.map((n) => n.domain).filter((d) => d && d !== 'core'))];
    list.replaceChildren(...domains.map((d) => {
      const li = document.createElement('li'), sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = this.domainColour(d);
      li.append(sw, d);
      return li;
    }));
  }

  layout() {
    const wrap = this.canvas.parentElement, w = wrap.clientWidth;
    if (!w) return;
    const bw = this.box.x1 - this.box.x0, bh = this.box.y1 - this.box.y0;
    const maxH = Math.max(260, Math.min(560, window.innerHeight * 0.7));
    this.size = Math.min(w / bw, maxH / bh);
    const cssW = Math.floor(bw * this.size), cssH = Math.floor(bh * this.size);
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    this.canvas.style.width = cssW + 'px';
    this.canvas.style.height = cssH + 'px';
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  // cell centre in CSS pixels
  px(n) { const [x, y] = centre(n.q, n.r); return [(x - this.box.x0) * this.size, (y - this.box.y0) * this.size]; }

  nodeAtEvent(e) {
    const b = this.canvas.getBoundingClientRect();
    const x = (e.clientX - b.left) / this.size + this.box.x0, y = (e.clientY - b.top) / this.size + this.box.y0;
    const [q, r] = cellAt(x, y);
    return this.at.get(key(q, r)) || null;
  }

  neighbours(n) { return n ? DIRS.map(([dq, dr]) => this.at.get(key(n.q + dq, n.r + dr))).filter(Boolean) : []; }

  hover(n) { if (n !== this.hovered) { this.hovered = n; this.update(); } }

  select(n, toggle = false) {
    this.selected = toggle && n === this.selected ? null : n;
    if (n) this.cursor = n;
    this.update();
  }

  key(e) {
    this.keyboard = true;
    const moves = { ArrowRight: DIRS[0], ArrowLeft: DIRS[3], ArrowUp: DIRS[2], ArrowDown: DIRS[5] };
    if (moves[e.key]) {
      const c = this.cursor || this.at.get(key(0, 0)), next = this.at.get(key(c.q + moves[e.key][0], c.r + moves[e.key][1]));
      if (next) this.cursor = next;
      this.update();
      e.preventDefault();
    } else if (e.key === 'Enter' || e.key === ' ') {
      if (this.cursor) this.select(this.cursor, true);
      e.preventDefault();
    } else if (e.key === 'Escape') {
      this.select(null);
    }
  }

  update() { this.draw(); this.read(); }

  hexPath(n, scale) {
    const [cx, cy] = this.px(n), s = this.size * scale, ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 180) * (60 * i - 90);
      const x = cx + s * Math.cos(a), y = cy + s * Math.sin(a);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.closePath();
  }

  draw() {
    const ctx = this.ctx, c = this.colour;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const near = new Set(this.neighbours(this.selected));
    const keyed = this.keyed();
    for (const n of this.nodes) {
      const isSel = n === this.selected, isNear = near.has(n), isHover = n === this.hovered;
      this.hexPath(n, 0.9);
      ctx.fillStyle = isSel ? c.accent : isHover ? c.hover : n.domain === 'core' ? c.cell : this.domainColour(n.domain);
      ctx.fill();
      if (isNear) { ctx.fillStyle = c.accentDim; ctx.fill(); }
      if (isSel || isNear || isHover) {
        ctx.lineWidth = isSel ? 1.5 : 1.25;
        ctx.strokeStyle = isNear ? c.accent : c.text;
        ctx.stroke();
      }
      if (n.domain === 'core' && !isSel) {                        // the origin: where every walk starts
        const [x, y] = this.px(n);
        ctx.beginPath();
        ctx.arc(x, y, Math.max(2, this.size * 0.22), 0, Math.PI * 2);
        ctx.fillStyle = c.accent;
        ctx.fill();
      }
    }
    if (keyed && this.cursor) {                                     // the keyboard's place, only while it is driving
      this.hexPath(this.cursor, 1.02);
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.text;
      ctx.stroke();
    }
  }

  // the panel beside the grid: what the cell under the pointer (or the keyboard, or the last click) holds
  keyed() { return this.keyboard && document.activeElement === this.canvas; }

  read() {
    const n = this.hovered || (this.keyed() && this.cursor) || this.selected;
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
    if (!n) {
      this.readout.replaceChildren(el('p', 'ro-hint', this.touch
        ? 'Tap a cell to read it and light up its neighbours.'
        : 'Hover a cell to read it. Click to light up its neighbours.'));
      return;
    }
    const head = el('p', 'ro-concept', n.concept || n.id);
    if (n.domain) head.append(' ', el('span', 'ro-domain', n.domain));
    const meta = el('dl', 'ro-meta');
    const row = (k, v) => { if (v == null || v === '') return; const d = el('div'); d.append(el('dt', null, k), el('dd', null, String(v))); meta.append(d); };
    row('cell', `q ${n.q}, r ${n.r}`);
    row('held by', Array.isArray(n.peers) ? n.peers.join(', ') : n.peers);
    row('recalled', n.recalls != null ? `${n.recalls} times` : null);
    row('stored', n.stored);
    const parts = [head, el('p', 'ro-chunk', n.chunk || ''), meta];
    if (n === this.selected) {
      const near = this.neighbours(n).map((m) => m.concept || m.id);
      const p = el('p', 'ro-near');
      p.append(el('span', 'ro-label', `${near.length} neighbours`), near.join(' · '));
      parts.push(p);
    } else if (this.selected && this.neighbours(this.selected).includes(n)) {
      parts.push(el('p', 'ro-near', `next to ${this.selected.concept || this.selected.id}`));
    }
    this.readout.replaceChildren(...parts);
  }
}

// ---------- start

const canvas = document.getElementById('field');
const readout = document.getElementById('readout');
if (canvas && readout) {
  loadNodes()
    .then((nodes) => {
      if (!nodes || !nodes.length) throw new Error('no nodes');
      const field = new Field(nodes, { canvas, readout, legend: document.getElementById('legend') });
      field.read();
      const count = document.getElementById('field-count');
      if (count) count.textContent = String(field.nodes.length);
    })
    .catch((err) => {
      canvas.hidden = true;
      readout.textContent = `The field could not load (${err.message}).`;
    });
}
