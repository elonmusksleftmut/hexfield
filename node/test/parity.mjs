// Does the JavaScript node give what p2p-hsdm's Python gives? Same inputs to both, compared.
//   PYTHONPATH=/path/to/p2p-hsdm node test/parity.mjs
// Vectors must match bit for bit; search results must be the same passages in the same order,
// with scores equal to within float32 rounding (numpy scores them in float32 through BLAS).

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { HsdmNode } from '../node.mjs';
import { ternarize } from '../hsdm.mjs';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const meta = JSON.parse(fs.readFileSync(path.join(ROOT, 'memory/memory.json'), 'utf8'));
const node = new HsdmNode({
  wasm: new WebAssembly.Module(fs.readFileSync(path.join(ROOT, 'core.wasm'))),
  vectors: fs.readFileSync(path.join(ROOT, 'memory/memory.bin')).buffer.slice(0),
  meta,
});

// a seeded generator, so a failure can be replayed
let seed = 20261121;
const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const pick = (a) => a[Math.floor(rand() * a.length)];

const passages = Array.from({ length: 250 }, () => pick(meta.texts));
const knownItem = Array.from({ length: 250 }, () => {
  const w = pick(meta.texts).split(/\s+/), s = Math.floor(rand() * Math.max(1, w.length - 14));
  return w.slice(s, s + 14).filter(() => rand() > 0.3).join(' ');
});
const odd = [
  '', 'the of and to a in', 'how do I parse json', 'read a csv file', 'Größe naïve café İstanbul KELVIN K',
  'x'.repeat(300), `${'a'.repeat(150)} ${'b'.repeat(150)} tail`, "don't won't o'clock rock'n'roll", '__init__ x_1 2048 0x7f',
  'zzzqqq unknownword', 'repeat repeat repeat repeat words words', meta.texts.slice(0, 6).join('\n\n'),
  'Socket SOCKET socket sockets', 'tab\tseparated\nnew lines\r\nand  spaces', '🙂 emoji 🙂 and text',
];
const encodeTexts = [...passages, ...knownItem, ...odd];
const questions = ['how do I parse json', 'read a csv file', 'send an email with an attachment', 'open a socket connection to a server',
  'regular expression match groups', 'run a shell command and capture its output', 'create a temporary file',
  'lock shared data between threads', 'round a decimal to two places', 'draw a circle on the screen', 'the of and',
  'zzzqqq', ...knownItem.slice(0, 150)];

// raw vectors for ternarize alone: real encodings, rescaled ones, and noise
const vecs = [];
for (let i = 0; i < 400; i++) {
  const acc = node.encoder.accumulate(encodeTexts[i % encodeTexts.length]);
  const scale = [1, 1e-3, 37.5, 2 ** -20][i % 4];
  vecs.push(acc.map((v) => v * scale));
}
for (let i = 0; i < 400; i++) {
  const a = new Float32Array(2048);
  for (let j = 0; j < 2048; j++) a[j] = rand() < 0.3 ? 0 : (rand() - 0.5) * (i % 7 + 1) ** 3;
  vecs.push(a);
}
const hexOf = (f) => Buffer.from(f.buffer, f.byteOffset, f.byteLength).toString('hex');

const job = { encode: encodeTexts, ternarize: vecs.map(hexOf), search: questions };
const py = JSON.parse(execFileSync('python3', [path.join(ROOT, 'test/parity.py')], {
  input: JSON.stringify(job), maxBuffer: 1 << 30,
}).toString());

let failures = 0;
const fail = (msg) => { failures++; if (failures <= 12) console.log('FAIL ' + msg); };
const tritHex = (t) => {
  const half = t.words.length / 2, pos = t.words.subarray(0, half), neg = t.words.subarray(half);
  return Buffer.concat([Buffer.from(pos.buffer, pos.byteOffset, pos.byteLength), Buffer.from(neg.buffer, neg.byteOffset, neg.byteLength)]).toString('hex');
};

encodeTexts.forEach((text, i) => {
  const acc = node.encoder.accumulate(text);
  if (hexOf(acc) !== py.encode[i].accum) fail(`accumulate differs for ${JSON.stringify(text.slice(0, 60))}`);
  if (tritHex(ternarize(acc)) !== py.encode[i].trits) fail(`encode differs for ${JSON.stringify(text.slice(0, 60))}`);
});
console.log(`${failures ? 'FAIL' : 'ok  '} ${encodeTexts.length} texts: accumulate and encode identical to p2p-hsdm, bit for bit`);
const f1 = failures;

vecs.forEach((v, i) => { if (tritHex(ternarize(v)) !== py.ternarize[i]) fail(`ternarize differs for vector ${i}`); });
console.log(`${failures > f1 ? 'FAIL' : 'ok  '} ${vecs.length} vectors: ternarize identical to numpy's (float32 pairwise standard deviation)`);
const f2 = failures;

// numpy picks among exactly tied scores in an order that depends on the CPU (argpartition is
// SIMD code), so a tie may resolve differently here. Those are told apart and counted, not failed:
// the candidates' scores must agree exactly, and results may differ only where scores tie.
let worst = 0, same = 0, tied = 0;
const tritScores = (q) => node.index.cosine(node.encoder.encode(q));
questions.forEach((q, i) => {
  const js = node.search(q, 5).results, ref = py.search[i].results;
  const scores = tritScores(q), cand = [...scores].sort((a, b) => b - a).slice(0, 5).filter((s) => s >= Math.fround(0.04));
  if (cand.join() !== py.search[i].candidates.map(Math.fround).join()) return fail(`search ${JSON.stringify(q)}: candidate scores differ`);
  if (js.length !== ref.length) return fail(`search ${JSON.stringify(q)}: ${js.length} results, python ${ref.length}`);
  const identical = js.every((r, k) => r.text === ref[k].text && r.label === ref[k].label);
  if (identical) { same++; js.forEach((r, k) => { worst = Math.max(worst, Math.abs(r.score - ref[k].score)); }); return; }
  // a tie: the candidate list's lowest score is shared with rows left out, or equal texts/scores swapped
  const cut = cand[cand.length - 1], atCut = scores.filter((s) => s === cut).length > cand.filter((s) => s === cut).length;
  const swapped = js.every((r, k) => r.text === ref[k].text || Math.abs(r.score - ref[k].score) < 1e-6);
  if (atCut || swapped) { tied++; return; }
  fail(`search ${JSON.stringify(q)}: ${js.map((r) => r.label).join(', ')} vs python ${ref.map((r) => r.label).join(', ')}`);
});
if (worst > 2e-6) fail(`scores differ by up to ${worst}`);
console.log(`${failures > f2 ? 'FAIL' : 'ok  '} ${questions.length} searches: ${same} identical to Node.search (scores within ${worst.toExponential(1)}), ${tied} differ only in how exact ties fall`);

if (failures) { console.log(`${failures} failures`); process.exit(1); }
console.log('parity: all checks passed');
