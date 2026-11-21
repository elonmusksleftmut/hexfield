// Hexfield: the field. A live HSDM node's memory on a hex grid, drawn on a canvas.
// Hover a cell to read what it holds; click or tap it to light up its nearest neighbours.
// Ask the node something and watch where the question lands and how far its answers are.
// Keyboard: arrow keys move between cells, Enter lights one up, Escape lets go.
//
// The grid comes in through loadNodes() and nowhere else. Each cell is
//   { id, q, r, concept, domain, modules, holds, chunk }
// where (q, r) are its axial coordinates. NODES_URL is the live node's grid; a JSON file with
// the same shape (or a bare array of cells) works too.

'use strict';

const NODE = 'https://node.hexfield.org';
const NODES_URL = `${NODE}/api/field`;

async function loadNodes() {
  const res = await fetch(NODES_URL, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`the node answered ${res.status}`);
  const data = await res.json();
  return Array.isArray(data) ? { nodes: data } : data;
}

async function askNode(question, k = 5) {
  const res = await fetch(`${NODE}/api/search?k=${k}&q=${encodeURIComponent(question)}`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`the node answered ${res.status}`);
  return res.json();
}

// ---------- hex arithmetic: pointy-top cells, axial coordinates

const SQRT3 = Math.sqrt(3);
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];
const key = (q, r) => `${q},${r}`;
const centre = (q, r) => [SQRT3 * (q + r / 2), 1.5 * r];          // in units of the cell's radius
const slug = (s) => String(s).toLowerCase().replace(/[^a-z]+/g, '-').replace(/^-|-$/g, '');

function cellAt(x, y) {                                            // the nearest cell to a point (same units)
  const qf = (SQRT3 / 3) * x - y / 3, rf = (2 / 3) * y, sf = -qf - rf;
  let q = Math.round(qf), r = Math.round(rf);
  const s = Math.round(sf), dq = Math.abs(q - qf), dr = Math.abs(r - rf), ds = Math.abs(s - sf);
  if (dq > dr && dq > ds) q = -r - s;
  else if (dr > ds) r = -q - s;
  return [q, r];
}

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const plural = (n, one, many = one + 's') => `${n.toLocaleString('en')} ${n === 1 ? one : many}`;
const fromLanding = (n) => (n === 0 ? 'where it landed' : `${plural(n, 'cell')} away`);
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const WEAK = 0.12;                                                 // below this, a match is mostly chance

// ---------- the field

class Field {
  constructor(cells, { canvas, readout, legend }) {
    this.canvas = canvas;
    this.readout = readout;
    this.ctx = canvas.getContext('2d');
    this.all = cells;                                              // in the node's order: a walk names cells by index
    this.nodes = cells.filter((n) => Number.isInteger(n.q) && Number.isInteger(n.r));
    this.at = new Map(this.nodes.map((n) => [key(n.q, n.r), n]));
    this.hovered = null;
    this.selected = null;
    this.cursor = null;                                            // the keyboard's cell
    this.keyboard = false;                                         // is the keyboard driving (not a pointer)?
    this.lastPointer = -Infinity;
    this.answer = null;                                            // the last question: where it landed, the walk, the answers
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
      if (!this.cursor) this.cursor = this.selected || this.answer?.land || this.at.get(key(0, 0)) || this.nodes[0];
      this.update();
    });
    canvas.addEventListener('blur', () => { this.update(); });
    new ResizeObserver(() => this.layout()).observe(canvas.parentElement);
    this.layout();
  }

  readColours() {
    const css = getComputedStyle(document.documentElement), v = (name) => css.getPropertyValue(name).trim();
    this.colour = {
      bg: v('--bg'), cell: v('--cell'), hover: v('--cell-hover'), text: v('--text'),
      accent: v('--accent'), accentDim: v('--accent-dim'), walk: v('--walk'),
    };
    this.font = v('--mono');
    this.domainColour = (d) => v(`--d-${slug(d)}`) || this.colour.cell;
  }

  drawLegend(list) {
    if (!list) return;
    const count = new Map();
    for (const n of this.nodes) if (n.domain) count.set(n.domain, (count.get(n.domain) || 0) + 1);
    list.replaceChildren(...[...count].sort((a, b) => b[1] - a[1]).map(([d]) => {
      const li = el('li'), sw = el('span', 'swatch');
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

  // ---------- a question: where it landed, the walk out from there, and where the answers are

  show(res) {
    cancelAnimationFrame(this.frame);
    const results = res.results || [], walk = res.walk;
    const answers = results.map((r, i) => ({ ...r, rank: i + 1, cellNode: this.at.get(key(r.q, r.r)), step: walk?.answers?.[i]?.step, away: walk?.answers?.[i]?.away }));
    const path = (walk?.path || []).map((i) => this.all[i]).filter(Boolean);
    const land = walk ? this.at.get(key(walk.land.q, walk.land.r)) : null;
    this.answer = { answers, path, land, shown: reducedMotion() ? path.length : 0 };
    this.selected = null;
    this.cursor = land || answers[0]?.cellNode || this.cursor;
    if (!path.length || reducedMotion()) { this.update(); return this.answer; }
    // the walk, step by step: about a second and a half however far it goes
    const per = Math.max(6, Math.min(60, 1500 / path.length)), start = performance.now();
    const tick = (now) => {
      this.answer.shown = Math.min(path.length, 1 + Math.floor((now - start) / per));
      this.draw();
      if (this.answer.shown < path.length) this.frame = requestAnimationFrame(tick);
      else this.update();
    };
    this.frame = requestAnimationFrame(tick);
    this.update();
    return this.answer;
  }

  clearAnswer() { cancelAnimationFrame(this.frame); this.answer = null; this.update(); }

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
    const ctx = this.ctx, c = this.colour, a = this.answer;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const near = new Set(this.neighbours(this.selected));
    const walked = new Set(a ? a.path.slice(0, a.shown) : []);
    const found = new Map();                                       // cell -> the best rank found in it so far
    if (a) for (const ans of a.answers) {
      if (!ans.cellNode || (a.path.length && !(ans.step < a.shown))) continue;
      if (!found.has(ans.cellNode)) found.set(ans.cellNode, ans.rank);
    }
    for (const n of this.nodes) {
      const isSel = n === this.selected, isNear = near.has(n), isHover = n === this.hovered, rank = found.get(n);
      this.hexPath(n, 0.9);
      ctx.fillStyle = isSel || rank ? c.accent : isHover ? c.hover : this.domainColour(n.domain);
      ctx.fill();
      if (walked.has(n) && !rank && !isSel) { ctx.fillStyle = c.walk; ctx.fill(); }
      if (isNear) { ctx.fillStyle = c.accentDim; ctx.fill(); }
      if (isSel || isNear || isHover) {
        ctx.lineWidth = isSel ? 1.5 : 1.25;
        ctx.strokeStyle = isNear ? c.accent : c.text;
        ctx.stroke();
      }
      if (rank) {
        const [x, y] = this.px(n);
        ctx.fillStyle = c.bg;
        ctx.font = `600 ${Math.max(9, Math.round(this.size * 0.85))}px ${this.font}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(rank), x, y + 0.5);
      }
    }
    if (a && a.land) {                                             // where the question landed
      this.hexPath(a.land, 1.08);
      ctx.lineWidth = 2.25;
      ctx.strokeStyle = c.text;
      ctx.stroke();
    }
    if (this.keyed() && this.cursor) {                             // the keyboard's place, only while it is driving
      this.hexPath(this.cursor, 1.02);
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.accent;
      ctx.stroke();
    }
  }

  // the panel beside the grid: what the cell under the pointer (or the keyboard, or the last click) holds,
  // else where the last question landed
  keyed() { return this.keyboard && document.activeElement === this.canvas; }

  read() {
    const n = this.hovered || (this.keyed() && this.cursor) || this.selected;
    if (!n && this.answer) return this.readAnswer();
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
    row('holds', n.holds != null ? plural(n.holds, 'passage') : null);
    row('from', Array.isArray(n.modules) ? n.modules.join(', ') : n.modules);
    row('cell', `q ${n.q}, r ${n.r}`);
    const parts = [head, el('p', 'ro-chunk', n.chunk || ''), meta];
    const mark = this.answer && this.answer.answers.filter((x) => x.cellNode === n).map((x) => x.rank);
    if (n === this.answer?.land) parts.push(el('p', 'ro-near', 'Your question landed here.'));
    if (mark && mark.length) parts.push(el('p', 'ro-near', `Holds answer ${mark.join(' and ')}.`));
    if (n === this.selected) {
      const p = el('p', 'ro-near');
      const near = this.neighbours(n).map((m) => m.concept || m.id);
      p.append(el('span', 'ro-label', `${near.length} neighbours`), near.join(' / '));
      parts.push(p);
    } else if (this.selected && this.neighbours(this.selected).includes(n)) {
      parts.push(el('p', 'ro-near', `next to ${this.selected.concept || this.selected.id}`));
    }
    this.readout.replaceChildren(...parts);
  }

  readAnswer() {
    const { land, answers } = this.answer, best = answers[0];
    if (!answers.length) { this.readout.replaceChildren(el('p', 'ro-hint', 'Nothing in this node matched those words.')); return; }
    if (!land) {
      this.readout.replaceChildren(el('p', 'ro-hint', 'Those are all common words, so the question lands nowhere in particular. The answers are numbered on the grid.'));
      return;
    }
    const head = el('p', 'ro-concept', land.concept);
    head.append(' ', el('span', 'ro-domain', 'landed here'));
    const parts = [head, el('p', 'ro-chunk', `Your question landed on the ringed cell, among ${(land.modules || []).join(', ')}. ` +
      (best.score < WEAK ? 'Nothing here matches it well.' : best.away === 0 ? 'The best answer is in that cell.' : `The best answer is ${fromLanding(best.away)}.`))];
    const list = el('ol', 'ro-list');
    for (const x of answers) {
      const li = el('li');
      li.append(el('span', 'ro-l', `${x.rank}. ${x.label}`), el('span', 'ro-w', x.away == null ? '' : fromLanding(x.away)));
      list.append(li);
    }
    parts.push(list);
    this.readout.replaceChildren(...parts);
  }
}

// ---------- answers, listed under the grid

function listAnswers(box, res, field) {
  const list = box.querySelector('ol'), status = box.querySelector('[role="status"]');
  const results = res.results || [], answers = field ? field.answer?.answers || [] : [];
  box.hidden = false;
  if (!results.length) {
    status.textContent = 'Nothing in this node matched those words. HSDM matches words, not meaning: try the words the docs would use.';
    list.replaceChildren();
    return;
  }
  const best = answers[0];
  if (results[0].score < WEAK) {
    status.textContent = 'Nothing here matches those words well: these are the nearest it has, and they are near by chance. HSDM matches words, not meaning, so try the words the docs would use.';
  } else {
    status.textContent = `${plural(results.length, 'answer')} from ${res.query ? `“${res.query}”` : 'the node'}` +
      (best && best.away != null ? `. The best was ${best.away === 0 ? 'in the cell your question landed on' : fromLanding(best.away) + ' from where your question landed'}.` : '.');
  }
  list.replaceChildren(...results.map((r, i) => {
    const li = el('li'), b = el('button', r.score < WEAK ? 'answer weak' : 'answer');
    b.type = 'button';
    const head = el('span', 'an-head');
    head.append(el('span', 'an-rank', String(i + 1)), el('span', 'an-label', r.label));
    const where = answers[i] && answers[i].away != null ? fromLanding(answers[i].away) : '';
    if (where) head.append(el('span', 'an-where', where));
    head.append(el('span', 'an-score', r.score.toFixed(2)));
    const text = r.text.replace(/\s+/g, ' ').trim();
    b.append(head, el('span', 'an-text', text.length > 360 ? text.slice(0, text.lastIndexOf(' ', 360)) + ' …' : text));
    const cell = field && field.at.get(key(r.q, r.r));
    if (cell) {
      b.addEventListener('mouseenter', () => field.hover(cell));
      b.addEventListener('mouseleave', () => field.hover(null));
      b.addEventListener('focus', () => field.hover(cell));
      b.addEventListener('blur', () => field.hover(null));
      b.addEventListener('click', () => field.select(cell));
    }
    li.append(b);
    return li;
  }));
}

// what the grid itself shows, measured by the node when it laid the grid out
function proof(p, node) {
  const m = node && node.metrics;
  if (!p || !m || !m.similarity_by_distance || !m.similarity_random) return;
  const rnd = m.similarity_random, d = m.similarity_by_distance, x = (v) => (v / rnd).toFixed(1).replace(/\.0$/, '');
  const fade = d.findIndex((v, i) => i > 0 && v != null && v / rnd < 1.15);
  let text = `Measured on this grid: two passages in the same cell are ${x(d[0])} times as alike as two picked at random, and neighbours ${x(d[1])} times`;
  text += fade > 1 ? `; ${fade} cells apart they are hardly more alike than chance.` : '.';
  const by5 = m.best_reached_by && m.best_reached_by['0.05'];
  if (by5 && m.walk_questions) text += ` For ${Math.round(by5 * 100)}% of ${m.walk_questions} test questions, the best answer was in the first 5% of the memory the walk crossed.`;
  p.textContent = text;
  p.hidden = false;
}

// ---------- start

const canvas = document.getElementById('field');
const readout = document.getElementById('readout');
const form = document.getElementById('ask');
const answersBox = document.getElementById('answers');
let field = null;

if (canvas && readout) {
  loadNodes()
    .then(({ nodes, node }) => {
      if (!nodes || !nodes.length) throw new Error('no cells');
      field = new Field(nodes, { canvas, readout, legend: document.getElementById('legend') });
      field.read();
      const count = document.getElementById('field-count');
      if (count && node && node.passages) count.textContent = node.passages.toLocaleString('en');
      proof(document.getElementById('field-proof'), node);
      if (form) form.hidden = false;
    })
    .catch((err) => {
      canvas.hidden = true;
      readout.textContent = `The node could not be reached (${err.message}). Try again in a minute.`;
    });
}

if (form && answersBox) {
  const input = form.querySelector('input'), button = form.querySelector('button[type="submit"]');
  const ask = async (q) => {
    q = q.trim();
    if (!q) { input.focus(); return; }
    button.disabled = true;
    form.setAttribute('aria-busy', 'true');
    answersBox.hidden = false;
    answersBox.querySelector('[role="status"]').textContent = 'Asking the node…';
    try {
      const res = await askNode(q.slice(0, 300));
      if (field) field.show(res);
      listAnswers(answersBox, res, field);
    } catch (err) {
      answersBox.querySelector('[role="status"]').textContent = `The node didn't answer (${err.message}). Try again.`;
      answersBox.querySelector('ol').replaceChildren();
    } finally {
      button.disabled = false;
      form.removeAttribute('aria-busy');
    }
  };
  form.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });
  for (const b of form.querySelectorAll('[data-ask]')) {
    b.addEventListener('click', () => { input.value = b.dataset.ask; ask(b.dataset.ask); });
  }
}
