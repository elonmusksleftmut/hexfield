// HSDM's encoder and search, ported from p2p-hsdm (hsdm/hdc.py, hsdm/index.py, hsdm/node.py)
// to JavaScript. It gives the same vectors as the Python, bit for bit: test/parity.mjs checks it.
// The two hot loops (keyed BLAKE2b, AND + popcount) are WebAssembly, in core.c.
//
// Where numpy does float32 arithmetic, this does too (Math.fround), in numpy's order: the
// quantisation threshold is a float32 standard deviation summed pairwise, the way numpy sums.

const f32 = Math.fround;

export const WORD_RE = /[a-z0-9_]+(?:'[a-z]+)?/g;

// Function words carry almost no topical signal; down-weighted, not dropped (as in hdc.py).
export const STOPWORDS = new Set(`a an and are as at be but by for from has have he her his i if in into is it its
  not of on or she so that the their them then there these they this to was we were
  what when which who will with you your can do does did been being than too very
  all any each no nor only own same such both more most other some up out about
  over under again once here why how where also may might must shall should would
  could just`.split(/\s+/));

export const FEATURE_NNZ = 32;                  // signed positions set by one feature
const MIN_SCORE = f32(0.04);                     // index.search's floor, compared in float32 as numpy does

export function tokenize(text) { return text.toLowerCase().match(WORD_RE) || []; }

// ---------- the WebAssembly core: memory, and a bump allocator over it

export class Core {
  constructor(module) {
    this.ex = new WebAssembly.Instance(module, {}).exports;
    this.top = (this.ex.__heap_base.value + 15) & ~15;
    this.text = new TextEncoder();
  }
  alloc(bytes) {
    const at = this.top;
    this.top = (at + bytes + 15) & ~15;
    const short = this.top - this.ex.memory.buffer.byteLength;
    if (short > 0) this.ex.memory.grow(Math.ceil(short / 65536));
    return at;
  }
  get u8() { return new Uint8Array(this.ex.memory.buffer); }
  i32(at, n) { return new Int32Array(this.ex.memory.buffer, at, n); }
}

// ---------- numpy's float32 pairwise summation (numpy/_core/src/umath/loops_utils.h.src)

function pairwise(a, off, n) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = f32(res + a[off + i]);
    return res;
  }
  if (n <= 128) {
    let r0 = a[off], r1 = a[off + 1], r2 = a[off + 2], r3 = a[off + 3];
    let r4 = a[off + 4], r5 = a[off + 5], r6 = a[off + 6], r7 = a[off + 7];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      r0 = f32(r0 + a[off + i]); r1 = f32(r1 + a[off + i + 1]);
      r2 = f32(r2 + a[off + i + 2]); r3 = f32(r3 + a[off + i + 3]);
      r4 = f32(r4 + a[off + i + 4]); r5 = f32(r5 + a[off + i + 5]);
      r6 = f32(r6 + a[off + i + 6]); r7 = f32(r7 + a[off + i + 7]);
    }
    let res = f32(f32(f32(r0 + r1) + f32(r2 + r3)) + f32(f32(r4 + r5) + f32(r6 + r7)));
    for (; i < n; i++) res = f32(res + a[off + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f32(pairwise(a, off, n2) + pairwise(a, off + n2, n - n2));
}

// ---------- ternary vectors: two bit masks, as Uint32 words (the same bytes as numpy's uint64)

export class Trit {
  constructor(dim) {
    this.dim = dim;
    this.words = new Uint32Array(dim / 16);       // positive mask, then negative mask
    this.nnz = 0;
  }
  get pos() { return this.words.subarray(0, this.dim / 32); }
  get neg() { return this.words.subarray(this.dim / 32); }
}

// hdc.ternarize for one vector: entries beyond zero_band standard deviations become +1 / -1
export function ternarize(acc, zeroBand = 0.6) {
  const n = acc.length, t = new Trit(n), half = n / 32;
  const mean = f32(pairwise(acc, 0, n) / n);
  const sq = new Float32Array(n);
  for (let i = 0; i < n; i++) { const d = f32(acc[i] - mean); sq[i] = f32(d * d); }
  const sigma = f32(Math.sqrt(f32(pairwise(sq, 0, n) / n)));
  const thr = f32(f32(zeroBand) * sigma), w = t.words;
  for (let i = 0; i < n; i++) {
    if (acc[i] > thr) { w[i >>> 5] |= 1 << (i & 31); t.nnz++; }
    else if (acc[i] < -thr) { w[half + (i >>> 5)] |= 1 << (i & 31); t.nnz++; }
  }
  return t;
}

export function toDense(t) {
  const out = new Int8Array(t.dim), half = t.dim / 32;
  for (let i = 0; i < t.dim; i++) {
    const b = 1 << (i & 31);
    out[i] = t.words[i >>> 5] & b ? 1 : t.words[half + (i >>> 5)] & b ? -1 : 0;
  }
  return out;
}

// Trits.cosine for one row: float32, as numpy computes it
export function tritCosine(dot, nnzRow, nnzQuery) {
  const denom = f32(Math.sqrt(nnzRow * nnzQuery));
  return denom > 0 ? f32(dot / denom) : 0;
}

// hdc.cosine_dense. numpy does this in float32 through BLAS, in an order BLAS picks;
// float64 here agrees with it to about one part in a million.
export function cosineDense(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return aa && bb ? ab / (Math.sqrt(aa) * Math.sqrt(bb)) : 0;
}

// ---------- the encoder (hdc.Encoder)

export class Encoder {
  constructor(core, swarmKey, dim = 2048, { cacheSize = 60000 } = {}) {
    if (dim % 64 || dim & (dim - 1) || dim < 256 || dim > 8192) throw new Error('dim must be a power of two between 256 and 8192');
    this.core = core;
    this.dim = dim;
    this.idf = new Map();
    this.defaultIdf = 1.0;
    this.cache = new Map();
    this.cacheSize = cacheSize;                    // Python keeps 400,000; a Worker has 128 MB
    this.cap = 1024;
    this.msg = core.alloc(this.cap);
    this.out = core.alloc(64);
    const key = core.alloc(32), seed = core.text.encode('hsdm-encoder');
    const sk = typeof swarmKey === 'string' ? core.text.encode(swarmKey) : swarmKey;
    const at = core.alloc(seed.length + sk.length), mem = core.u8;
    mem.set(seed, at);
    mem.set(sk, at + seed.length);
    core.ex.hash(at, seed.length + sk.length, 32, this.out);
    core.u8.copyWithin(key, this.out, this.out + 32);
    core.ex.init(key);
  }

  // views into wasm memory, made again only if the memory grew (which detaches the old ones)
  _views() {
    const buf = this.core.ex.memory.buffer;
    if (this._buf !== buf) {
      this._buf = buf;
      this._msg = new Uint8Array(buf, this.msg, this.cap);
      this._out = new Uint16Array(buf, this.out, 32);
    }
  }

  _write(text) {
    if (text.length * 3 > this.cap) { this.cap = text.length * 3; this.msg = this.core.alloc(this.cap); this._buf = null; }
    this._views();
    return this.core.text.encodeInto(text, this._msg).written;
  }

  // a feature's 32 signed positions, as the raw little-endian uint16s of its BLAKE2b digest
  feature(feat) {
    let hit = this.cache.get(feat);
    if (!hit) {
      this.core.ex.hash_feature(this.msg, this._write(feat), this.out);
      hit = this._out.slice();
      if (this.cache.size < this.cacheSize) this.cache.set(feat, hit);
    }
    return hit;
  }

  // keyed hash of a word into a vocabulary-sketch bucket (buckets must be a power of two up to 2^32)
  bucket(word, buckets) { return (this.core.ex.hash_vocab(this.msg, this._write(word)) >>> 0) % buckets; }

  fitIdf(tokenLists) {
    const df = new Map();
    let n = 0;
    for (const toks of tokenLists) { for (const w of new Set(toks)) df.set(w, (df.get(w) || 0) + 1); n++; }
    if (n < 20) { this.idf = new Map(); this.defaultIdf = 1.0; return; }
    this.idf = new Map([...df].map(([w, c]) => [w, Math.log(1.0 + (n - c + 0.5) / (c + 0.5))]));
    this.defaultIdf = Math.log(1.0 + (n + 0.5) / 0.5);
  }

  weight(word) {
    const w = this.idf.has(word) ? this.idf.get(word) : this.defaultIdf;
    return STOPWORDS.has(word) ? w * 0.15 : w;
  }

  features(tokens) {
    const feats = new Map(), counts = new Map();
    for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
    for (const [word, tf] of counts) {
      const w = this.weight(word) * (1.0 + Math.log(tf));
      feats.set('w:' + word, w);
      if (word.length >= 5 && /^[a-z]+$/.test(word)) {          // light morphological smoothing
        const padded = `<${word}>`, n = padded.length - 2;
        for (let i = 0; i < n; i++) {
          const key = 't:' + padded.slice(i, i + 3);
          feats.set(key, (feats.get(key) || 0.0) + 0.15 * w / Math.sqrt(n));
        }
      }
    }
    for (let i = 0; i + 1 < tokens.length; i++) {               // word order
      const key = `b:${tokens[i]} ${tokens[i + 1]}`;
      if (!feats.has(key)) feats.set(key, 0.5 * Math.sqrt(this.weight(tokens[i]) * this.weight(tokens[i + 1])));
    }
    return feats;
  }

  // the real-valued hypervector of a text: float64 sums of float32 weights, as np.bincount does,
  // then float32. (Those sums are exact, so the order features arrive in can't change them.)
  accumulate(textOrTokens) {
    const tokens = typeof textOrTokens === 'string' ? tokenize(textOrTokens) : textOrTokens;
    const acc = new Float64Array(this.dim), mask = this.dim - 1;
    for (const [feat, weight] of this.features(tokens)) {
      const v = this.feature(feat), w = f32(weight);
      for (let j = 0; j < FEATURE_NNZ; j++) acc[v[j] & mask] += v[j] & 0x8000 ? w : -w;
    }
    return Float32Array.from(acc);
  }

  encode(textOrTokens) { return ternarize(this.accumulate(textOrTokens)); }
}

// ---------- index.split_passages

export const MAX_PASSAGE_BYTES = 900;

export function splitPassages(text, maxWords = 110, overlap = 20) {
  const out = [];
  let buf = [], bufWords = 0;
  const flush = () => { if (buf.length) out.push(buf.join('\n')); buf = []; bufWords = 0; };
  for (let para of text.split(/\n\s*\n/)) {
    para = para.trim();                                          // (Python strips the same whitespace for ASCII text)
    if (!para) continue;
    const words = para.split(/\s+/);
    if (words.length > maxWords) {
      flush();
      const step = maxWords - overlap;
      for (let i = 0; i < Math.max(1, words.length - overlap); i += step) out.push(words.slice(i, i + maxWords).join(' '));
      continue;
    }
    if (bufWords + words.length > maxWords) flush();
    buf.push(para);
    bufWords += words.length;
  }
  flush();
  const enc = new TextEncoder(), dec = new TextDecoder('utf-8', { fatal: false });
  return out.map((p) => {
    const raw = enc.encode(p);
    if (raw.length <= MAX_PASSAGE_BYTES) return p;
    let cut = MAX_PASSAGE_BYTES;                                 // decode(errors="ignore"): drop a split character
    while (cut > 0 && (raw[cut] & 0xc0) === 0x80) cut--;
    return dec.decode(raw.subarray(0, cut));
  }).filter((p) => tokenize(p).length >= 3);
}

// ---------- a node's local index: packed vectors in wasm memory, searched by AND + popcount

export class Index {
  // raw: Python's Trits.to_bytes() for `rows` vectors (all positive masks, then all negative masks)
  constructor(core, raw, rows, dim = 2048) {
    this.core = core;
    this.rows = rows;
    this.dim = dim;
    this.words = dim / 64;                         // uint64 words per mask
    const bytes = rows * this.words * 8;
    if (raw.byteLength !== 2 * bytes) throw new Error('trit payload has the wrong length');
    this.pos = core.alloc(2 * bytes);
    this.neg = this.pos + bytes;
    core.u8.set(new Uint8Array(raw), this.pos);
    const nnz = core.alloc(rows * 4);
    core.ex.nnz(this.pos, this.neg, rows, this.words, nnz);
    this.nnz = Float32Array.from(core.i32(nnz, rows));
    this.q = core.alloc(dim / 4);
    this.dots = core.alloc(rows * 4);
    this.sel = core.alloc(rows * 4);
  }

  _query(t) { this.core.u8.set(new Uint8Array(t.words.buffer, t.words.byteOffset, t.words.byteLength), this.q); }

  // Trits.cosine: every row against one query
  cosine(t) {
    this._query(t);
    this.core.ex.scan(this.pos, this.neg, this.rows, this.words, this.q, this.dots);
    const dots = this.core.i32(this.dots, this.rows), out = new Float32Array(this.rows);
    for (let i = 0; i < this.rows; i++) out[i] = tritCosine(dots[i], this.nnz[i], t.nnz);
    return out;
  }

  // the same for some rows only
  cosineRows(t, rows) {
    this._query(t);
    new Uint32Array(this.core.ex.memory.buffer, this.sel, rows.length).set(rows);
    this.core.ex.scan_rows(this.pos, this.neg, this.sel, rows.length, this.words, this.q, this.dots);
    const dots = this.core.i32(this.dots, rows.length), out = new Float32Array(rows.length);
    for (let k = 0; k < rows.length; k++) out[k] = tritCosine(dots[k], this.nnz[rows[k]], t.nnz);
    return out;
  }
}

// LocalIndex.search: the best `limit` rows by trit cosine (ties to the lower row), above the floor
export function topRows(scores, limit) {
  const top = [];
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    if (top.length === limit) {
      if (!(s > scores[top[limit - 1]])) continue;
      top.pop();
    }
    let j = top.length;
    top.push(i);
    while (j > 0 && scores[top[j - 1]] < s) { top[j] = top[j - 1]; j--; }
    top[j] = i;
  }
  return top.filter((i) => scores[i] >= MIN_SCORE).map((i) => ({ row: i, score: scores[i] }));
}

// Node.search on a node with no peers: the index's best candidates, re-scored against the
// full-precision query the way the asker fuses every peer's answers.
export function search(encoder, index, passages, text, limit = 5) {
  const accum = encoder.accumulate(text), query = ternarize(accum);
  const perPeer = Math.min(8, Math.max(limit, 3));
  const found = topRows(index.cosine(query), perPeer);
  const seen = new Set(), results = [];
  for (const { row } of found) {
    const p = passages(row);
    if (seen.has(p.text)) continue;
    seen.add(p.text);
    results.push({ row, score: cosineDense(accum, encoder.accumulate(p.text)), label: p.label, text: p.text });
  }
  results.sort((a, b) => b.score - a.score);
  return { accum, query, results: results.slice(0, limit) };
}
