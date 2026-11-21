"""Builds the hexfield node's memory, with p2p-hsdm's own Python code.

The memory is the Python standard library's documentation: every docstring of 12 words or
more, from modules with at least three of them (the rules of `hsdm bench`, minus numpy, plus the
smaller modules). p2p-hsdm's LocalIndex splits and encodes it, so the vectors are exactly what a
Python node holding the same text would hold.

Then the memory is laid out on the 217-cell hex grid with a self-organising map: each cell gets
a prototype vector, and every passage goes to the cell whose prototype it is most like, each cell
taking an even share. Nothing places it by hand, so whether neighbours are related is a measured
property, and the script measures it.

    PYTHONPATH=/path/to/p2p-hsdm python3 build.py      # writes memory/memory.bin, memory/memory.json
"""

from __future__ import annotations

import ast
import datetime
import heapq
import json
import math
import os
import platform
import random
import re
import sys
import sysconfig
from collections import Counter

import numpy as np

from hsdm.hdc import STOPWORDS, Trits, cosine_dense, ternarize, tokenize
from hsdm.index import LocalIndex
from hsdm.node import Node

KEY = b"hexfield public swarm"  # public on purpose: everything this node holds is public
RADIUS = 8                      # 217 cells
HERE = os.path.dirname(os.path.abspath(__file__))

# The library reference groups modules by what they are for; these are those groups, merged to nine.
FAMILIES = {
    "text": "re difflib textwrap codecs locale string shlex unicodedata gettext stringprep",
    "data": "email xml json csv configparser tarfile zipfile bz2 lzma gzip pickle pickletools copyreg shelve dbm "
            "sqlite3 plistlib mailbox mimetypes base64 quopri html wave marshal binascii zlib",
    "numbers & time": "decimal statistics fractions numbers random pylong cmath math datetime calendar strptime sched zoneinfo",
    "types": "collections typing enum functools dataclasses weakref copy heapq bisect operator types "
             "graphlib abc pprint contextlib reprlib",
    "system": "os pathlib io shutil tempfile glob fileinput filecmp linecache ntpath posixpath genericpath nturl2path stat "
              "platform subprocess pty getpass curses ctypes osx_support sysconfig argparse optparse getopt cmd logging fnmatch",
    "network": "http urllib xmlrpc smtplib imaplib ftplib poplib socket socketserver ssl wsgiref ipaddress webbrowser uuid "
               "hmac hashlib secrets",
    "concurrency": "asyncio threading multiprocessing concurrent queue selectors",
    "language": "importlib unittest inspect doctest pdb bdb ast dis tokenize traceback pydoc code codeop warnings timeit trace "
                "profile pstats tracemalloc compileall py_compile pkgutil modulefinder runpy symtable pyclbr tabnanny "
                "__future__ zipimport site sitecustomize venv ensurepip pyrepl rlcompleter",
    "graphics": "turtle tkinter",
}
FAMILY_OF = {m: fam for fam, mods in FAMILIES.items() for m in mods.split()}


RENAME = {"_pydecimal": "decimal", "_pyio": "io", "_pydatetime": "datetime", "_py_abc": "abc",
          "_collections_abc": "collections.abc"}  # pure-Python twins, by the names people know


def display_name(module: str) -> str:
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    head, _, rest = module.partition(".")
    head = RENAME.get(head, head)
    return head + ("." + rest if rest else "")


def family(name: str) -> str:
    head = name.split(".")[0]
    for key in (head, head.lstrip("_")):
        if key in FAMILY_OF:
            return FAMILY_OF[key]
    raise SystemExit(f"no family for module {name!r}: add it to FAMILIES")


def load_stdlib(min_docs: int = 3) -> dict:
    """{module: [docstring]} for the standard library, by hsdm bench's rules."""
    root = sysconfig.get_paths()["stdlib"]
    docs = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {"test", "tests", "__pycache__", "idlelib", "site-packages", "dist-packages"})
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            module = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
            try:
                tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
            except (SyntaxError, ValueError):
                continue
            strings = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node)
                    if doc and len(doc.split()) >= 12:
                        strings.append(doc)
            if len(strings) >= min_docs:
                docs[module] = strings
    return {m: docs[m] for m in sorted(docs)}


# ---------------------------------------------------------------- the grid

def grid(radius: int):
    cells = [(q, r) for r in range(-radius, radius + 1) for q in range(-radius, radius + 1) if max(abs(q), abs(r), abs(q + r)) <= radius]
    qr = np.array(cells)
    dq = qr[:, None, 0] - qr[None, :, 0]
    dr = qr[:, None, 1] - qr[None, :, 1]
    dist = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2
    xy = np.stack([np.sqrt(3) * (qr[:, 0] + qr[:, 1] / 2), 1.5 * qr[:, 1]], axis=1)
    return cells, dist, xy


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def balanced(S: np.ndarray, cap: int) -> np.ndarray:
    """Each passage, the most decided first, takes the best cell that still has room."""
    out = np.empty(len(S), dtype=np.int64)
    load = np.zeros(S.shape[1], dtype=np.int64)
    for i in np.argsort(-S.max(axis=1), kind="stable"):
        c = int(np.argmax(np.where(load < cap, S[i], -np.inf)))
        out[i] = c
        load[c] += 1
    return out


def train_som(X: np.ndarray, dist: np.ndarray, xy: np.ndarray, epochs: int = 60, sigma0: float = 4.0, sigma1: float = 0.5):
    """Batch self-organising map on the unit sphere (cosine, as HSDM scores), from a PCA start.
    Cells share the memory evenly (a cell takes at most its share), so a large topic spreads
    over a region of neighbouring cells instead of crowding into one."""
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    scores = (X - mu) @ vt[:2].T
    span = xy / np.abs(xy).max(axis=0)
    W = unit(mu + 2 * span[:, :1] * scores[:, 0].std() * vt[0] + 2 * span[:, 1:] * scores[:, 1].std() * vt[1]).astype(np.float32)
    cap = int(np.ceil(len(X) / len(W)))
    d2 = dist.astype(np.float64) ** 2
    for e in range(epochs):
        sigma = sigma0 * (sigma1 / sigma0) ** (e / (epochs - 1))
        bmu = balanced(X @ W.T, cap)
        h = np.exp(-d2 / (2 * sigma * sigma))
        onehot = np.zeros((len(X), len(W)), dtype=np.float32)
        onehot[np.arange(len(X)), bmu] = 1
        W = unit((h @ (onehot.T @ X)) / np.maximum(h @ onehot.sum(axis=0), 1e-9)[:, None]).astype(np.float32)
    return W, balanced(X @ W.T, cap)


# ---------------------------------------------------------------- labels

def cell_terms(token_lists, members, idf, k=3):
    """The words that set a cell apart: in many of its passages, rare in the rest."""
    df = Counter()
    for i in members:
        df.update({w for w in token_lists[i] if len(w) >= 3 and w not in STOPWORDS and not w.isdigit() and not w.startswith("_")})
    scored = sorted(df, key=lambda w: (-df[w] * idf.get(w, 0.0), w))
    out, stems = [], set()
    for w in scored:
        stem = re.sub(r"(ing|ed|es|s)$", "", w)
        if stem in stems or any(stem.startswith(s) or s.startswith(stem) for s in stems if min(len(s), len(stem)) >= 5):
            continue
        out.append(w)
        stems.add(stem)
        if len(out) == k:
            break
    return out


def snippet(text: str, n: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: flat.rfind(" ", 0, n)] + " …"


# ---------------------------------------------------------------- landing and walking

def cell_vocab(token_sets, of, ncells):
    """For each cell, how many of its passages contain each word: the cell's own sketch."""
    df = [Counter() for _ in range(ncells)]
    for i, words in enumerate(token_sets):
        df[of[i]].update(words)
    return df


def route_cells(df, sizes, query, smoothing=0.2):
    """hsdm.index.route_scores with every cell standing in for a peer, and exact counts in place
    of the 4-bit sketch: how likely is each cell's text to produce the question's words, rarer
    words counting more. node.mjs computes the same."""
    words = sorted({w for w in tokenize(query) if w not in STOPWORDS})
    if not words:
        return None
    counts = np.array([[df[c].get(w, 0) for w in words] for c in range(len(df))], dtype=np.float64)
    sz = np.maximum(sizes.astype(np.float64), 1.0)
    total = counts.sum(axis=0)
    background = total / sz.sum()
    local = counts / sz[:, None]
    rarity = np.log1p(sz.sum() / np.maximum(total, 1.0))
    return (rarity * np.log((1 - smoothing) * local + smoothing * background + 1e-9)).sum(axis=1)


def walk_order(scores, start, neighbours):
    """Best first across the grid: from the landing cell, always step into the unvisited cell next
    to the walk so far that scores highest (ties to the lower index)."""
    seen, frontier, order = {start}, [(-scores[start], start)], []
    while frontier:
        _, c = heapq.heappop(frontier)
        order.append(c)
        for m in neighbours[c]:
            if m not in seen:
                seen.add(m)
                heapq.heappush(frontier, (-scores[m], int(m)))
    return order


# ---------------------------------------------------------------- build

def main() -> None:
    rng = random.Random(7)
    corpus = load_stdlib()
    node = Node(KEY)
    index = node.index
    for module, strings in corpus.items():
        for doc in strings:
            index.add_text(display_name(module), doc)
    index.build()
    passages = index.passages
    n = len(passages)
    labels = sorted({p.label for p in passages})
    for name in labels:
        family(name)
    print(f"{len(corpus)} modules, {n} passages, {sum(len(p.text.encode()) for p in passages) // 1024} KB of text")

    # the field
    cells, dist, xy = grid(RADIUS)
    # A passage is laid out with its module as context (its vector plus the module's mean), so a
    # module's passages stay together and modules that say similar things sit side by side.
    # Search never sees this: it is HSDM's, on the passage alone.
    X = unit(index.vectors.to_dense().astype(np.float32))
    label_of = np.array([labels.index(p.label) for p in passages])
    centroid = np.zeros((len(labels), X.shape[1]), dtype=np.float32)
    np.add.at(centroid, label_of, X)
    W, of = train_som(unit(X + unit(centroid)[label_of]).astype(np.float32), dist, xy)
    tokens = [tokenize(p.text) for p in passages]
    members = [np.flatnonzero(of == c) for c in range(len(cells))]
    sizes = np.array([len(m) for m in members])
    sims = X @ W.T

    field = []
    for c, (q, r) in enumerate(cells):
        own = members[c]
        near = own if len(own) >= 2 else np.flatnonzero(dist[c][of] <= 1)
        mods = Counter(passages[i].label for i in near)
        fams = Counter(family(passages[i].label) for i in near)
        rep = int(near[np.argmax(sims[near, c])])
        field.append({
            "id": f"c{c}", "q": q, "r": r,
            "concept": " · ".join(cell_terms(tokens, near, node.encoder.idf)),
            "domain": fams.most_common(1)[0][0],
            "modules": [m for m, _ in mods.most_common(3)],
            "holds": int(len(own)),
            "row": rep,
            "chunk": snippet(passages[rep].text),
        })

    # how alike are two passages, by how far apart their cells are? (the claim the page makes)
    a = np.array([rng.randrange(n) for _ in range(200_000)])
    b = np.array([rng.randrange(n) for _ in range(200_000)])
    keep = a != b
    a, b = a[keep], b[keep]
    cos = np.einsum("ij,ij->i", X[a], X[b])
    d = dist[of[a], of[b]]
    by_distance = [round(float(cos[d == k].mean()), 4) if np.any(d == k) else None for k in range(2 * RADIUS + 1)]
    random_pairs = round(float(cos.mean()), 4)

    # Where are the answers, relative to where the question lands? Known-item questions, the way
    # hsdm bench asks them (14 words of a passage, 30% deleted); the answers are what the node
    # returns, and the walk is the one node.mjs takes. How much memory has it crossed on reaching them?
    df = cell_vocab([set(t) for t in tokens], of, len(cells))
    neighbours = [np.flatnonzero(dist[c] == 1) for c in range(len(cells))]
    reach_best, reach_all = [], []
    for _ in range(800):
        words = passages[rng.randrange(n)].text.split()
        if len(words) < 30:
            continue
        s = rng.randrange(len(words) - 14)
        query = " ".join(w for w in words[s : s + 14] if rng.random() > 0.3)
        acc = node.encoder.accumulate(query)
        scores = index.vectors.cosine(Trits.from_dense(ternarize(acc)))
        top = [i for i in np.argsort(-scores, kind="stable")[:5] if scores[i] >= np.float32(0.04)]
        route = route_cells(df, sizes, query)
        if not top or route is None:
            continue
        top.sort(key=lambda i: -cosine_dense(acc, node.encoder.accumulate(passages[i].text)))
        order = walk_order(route, int(np.argmax(route)), neighbours)
        crossed = np.cumsum(sizes[order]) / n
        at = {c: k for k, c in enumerate(order)}
        reach_best.append(float(crossed[at[of[top[0]]]]))
        reach_all.append(float(max(crossed[at[of[i]]] for i in top)))
    reach_best, reach_all = np.array(reach_best), np.array(reach_all)

    metrics = {
        "similarity_by_distance": by_distance,
        "similarity_random": random_pairs,
        "walk_questions": len(reach_best),
        "best_reached_by": {f"{b:.2f}": round(float(np.mean(reach_best <= b)), 3) for b in (0.05, 0.1, 0.25, 0.5)},
        "all_reached_by": {f"{b:.2f}": round(float(np.mean(reach_all <= b)), 3) for b in (0.05, 0.1, 0.25, 0.5)},
        "best_reached_median": round(float(np.median(reach_best)), 4),
        "largest_cell": int(sizes.max()),
        "empty_cells": int((sizes == 0).sum()),
    }

    py = platform.python_version()
    meta = {
        "format": 1,
        "built": datetime.date.today().isoformat(),
        "key": KEY.decode(),
        "dim": node.encoder.dim,
        "rows": n,
        "radius": RADIUS,
        "corpus": {
            "name": f"the Python {py} standard library's docstrings",
            "python": py,
            "modules": len(corpus),
            "license": "Python Software Foundation License; the docstrings' own words, unedited",
        },
        "default_idf": node.encoder.default_idf,
        "idf": node.encoder.idf,
        "labels": labels,
        "label_of": [int(l) for l in label_of],
        "texts": [p.text for p in passages],
        "cell_of": [int(c) for c in of],
        "families": list(FAMILIES),
        "field": field,
        "metrics": metrics,
    }
    os.makedirs(os.path.join(HERE, "memory"), exist_ok=True)
    with open(os.path.join(HERE, "memory", "memory.bin"), "wb") as fh:
        fh.write(index.vectors.to_bytes())                       # exactly a Python node's vectors
    with open(os.path.join(HERE, "memory", "memory.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"field: {len(cells)} cells, {metrics['empty_cells']} empty, largest holds {metrics['largest_cell']}")
    print("mean similarity of two passages, by how many cells apart they are:")
    print("  " + "  ".join(f"{k}:{v}" for k, v in enumerate(by_distance) if v is not None) + f"   random pairs: {random_pairs}")
    print(f"walks ({metrics['walk_questions']} known-item questions): best answer reached having crossed "
          + ", ".join(f"{float(b):.0%} of memory: {v:.0%}" for b, v in metrics["best_reached_by"].items())
          + f" (median {metrics['best_reached_median']:.1%})")
    print("  all five answers: " + ", ".join(f"{float(b):.0%}: {v:.0%}" for b, v in metrics["all_reached_by"].items()))


if __name__ == "__main__":
    main()
