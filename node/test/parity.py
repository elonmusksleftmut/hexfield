"""The Python half of test/parity.mjs: p2p-hsdm's own answers for the same inputs.

Reads a JSON job on stdin, loads the node's memory into a real hsdm Node, and prints
p2p-hsdm's encodings and search results. Needs p2p-hsdm on PYTHONPATH.
"""

import asyncio
import json
import os
import sys

import numpy as np

from hsdm.hdc import Trits, ternarize
from hsdm.index import Passage
from hsdm.node import Node

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    meta = json.load(open(os.path.join(HERE, "memory", "memory.json"), encoding="utf-8"))
    raw = open(os.path.join(HERE, "memory", "memory.bin"), "rb").read()
    node = Node(meta["key"].encode(), dim=meta["dim"])
    node.encoder.idf, node.encoder.default_idf = meta["idf"], meta["default_idf"]
    node.index.passages = [Passage(meta["labels"][l], t) for l, t in zip(meta["label_of"], meta["texts"])]
    node.index.vectors = Trits.from_bytes(raw, meta["dim"], meta["rows"])
    return node


async def main():
    job = json.loads(sys.stdin.read())
    node = load()
    out = {"encode": [], "ternarize": [], "search": []}
    for text in job.get("encode", []):
        acc = node.encoder.accumulate(text)
        out["encode"].append({"accum": acc.astype("<f4").tobytes().hex(), "trits": node.encoder.encode(text).to_bytes().hex()})
    for hexed in job.get("ternarize", []):
        acc = np.frombuffer(bytes.fromhex(hexed), dtype="<f4")
        out["ternarize"].append(Trits.from_dense(ternarize(acc)).to_bytes().hex())
    for q in job.get("search", []):
        rep = await node.search(q, limit=5, fanout=3)
        found = node.index.search(Trits.from_dense(ternarize(node.encoder.accumulate(q))), 5)
        out["search"].append({
            "candidates": [float(score) for score, _ in found],   # trit cosines, before re-scoring
            "results": [{"score": r.score, "label": r.label, "text": r.text} for r in rep.results],
        })
    print(json.dumps(out))


asyncio.run(main())
