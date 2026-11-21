# node.hexfield.org

One HSDM node, running on Cloudflare Workers. It holds 4,162 passages: the docstrings of the
Python 3.13 standard library. The field on hexfield.org is this node's memory. Ask it something
there, or here:

```
curl 'https://node.hexfield.org/api/search?q=how+do+I+read+a+csv+file&k=3'
curl https://node.hexfield.org/api/status
curl https://node.hexfield.org/api/field
```

It is read-only and public, and it keeps no state: nothing you ask is stored.

## What it is

The node runs a JavaScript port of [p2p-hsdm](https://github.com/elonmusksleftmut/P2P-HSDM),
`hsdm.mjs`, with the two hot loops (keyed BLAKE2b and the AND + popcount scan) compiled to
WebAssembly from `core.c`. `build.py` builds the memory with p2p-hsdm's own Python code, so the
vectors the node holds are the ones a Python node would hold.

`/api/search` answers the way `Node.search` does on a node with no peers. It ranks every
passage by trit cosine, keeps the best five, re-scores them against the full-precision query and
sorts them. The JSON has the same fields as `hsdm/web.py`, plus each answer's cell and a `walk`.

## Checked against the Python

`PYTHONPATH=/path/to/p2p-hsdm node test/parity.mjs` feeds both implementations the same inputs:

- 515 texts, including passages, partial questions, Unicode, empty text and very long words. The
  hypervectors (`accumulate`) and their trits (`encode`) are identical, bit for bit.
- 800 vectors through `ternarize`, the same as numpy's. That includes the float32 standard
  deviation, which numpy sums pairwise.
- 162 searches. The candidates' scores agree exactly. 149 give the same passages in the same
  order, with scores within 5e-7. The other 13 differ only in how exact ties fall, which is the
  one thing numpy leaves to the CPU: its `argpartition` is SIMD code.

## The grid

`build.py` lays the memory out on 217 hex cells with a self-organising map trained on the HSDM
vectors. Each passage is placed with its module as context, and each cell takes an even share of
about 19 passages. Nothing is placed by hand, so whether neighbours are related is a measurement.
From the current build:

| cells apart | 0 | 1 | 2 | 4 | 8 | any two passages |
|---|---|---|---|---|---|---|
| mean similarity of two passages | 0.086 | 0.029 | 0.019 | 0.014 | 0.012 | 0.013 |

A question lands on a cell the way HSDM routes it to a peer. Every cell stands in for a peer,
scored by `route_scores` from its passages' words, with rarer words counting more. The walk then
steps out from that cell, best first. The field draws how many cells away each answer was. On
461 known-item questions (14 words of a passage with 30% deleted, as `hsdm bench` asks them),
the best answer was found by the time the walk had crossed:

| share of the memory crossed | 0.5% (median) | 5% | 10% | 25% |
|---|---|---|---|---|
| questions whose best answer was found | half | 91% | 93% | 96% |

The other four answers are usually weaker matches on one or two words, and they sit wherever
those words do. The node always answers from the full search; the walk shows where the answers
are on the grid.

It matches words, not meaning, like every HSDM node (see p2p-hsdm's AUDIT.md).

## Cost

One search takes about 1.5 ms of CPU, warm, in V8, the engine Workers run. That is inside the
free plan's 10 ms. The upload is 1.8 MB gzipped, and the plan allows 3 MB.

## Rebuild and deploy

```
PYTHONPATH=/path/to/p2p-hsdm python3 build.py    # memory/memory.bin and memory/memory.json
./build-wasm.sh                                  # core.wasm, if core.c changed (clang, wasm-ld)
PYTHONPATH=/path/to/p2p-hsdm node test/parity.mjs
npx wrangler deploy
```

The swarm key is public (`hexfield public swarm`), because everything the node holds is public.

## Files

```
core.c, core.wasm   BLAKE2b and the popcount scan, for WebAssembly
hsdm.mjs            the port: tokenizer, encoder, ternary vectors, index, Node.search
node.mjs            the node: search, the grid, landing and walking, status
worker.mjs          the Worker: GET /api/status, /api/search, /api/field
build.py            builds the memory with p2p-hsdm, and lays it out on the grid
memory/             what the node holds
test/parity.mjs     JavaScript against Python, for the same inputs
```
