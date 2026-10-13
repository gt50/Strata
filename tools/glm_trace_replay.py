"""Replays a GLM-5.3-Flash expert trace through expert-cache slot policies and prints the misses, split into decode
calls and prompt rounds.

    STRATA_GLM_EXPERT_TRACE=a.bin  (the engine appends every expert-cache call: int32 block, int32 kind - 0 decode,
                                    1 a prompt round - int32 k, k int32 expert ids)
    python tools/glm_trace_replay.py TEST.bin [--profile TRAIN.bin] [--slots 4874]

Policies: the cache's own (one LRU over every block's slots, filled at start with an equal share of every block),
the same without the MTP block's prompt calls (the engine skips them without drafts), per-block LRU partitions (equal,
and sized greedily from TRAIN's per-block miss curves), and Belady's optimum (evicting the expert whose next use is
furthest: what no cache can do, the bound).  A call's own experts are never evicted while it runs; a block's
partition may grow for one prompt round and shrinks back after it.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import struct
from collections import OrderedDict

NE = 288   # GLM-5.3-Flash's experts per block


def read(path):
    data = open(path, "rb").read()
    calls, o = [], 0
    while o + 12 <= len(data):
        il, kind, k = struct.unpack_from("<3i", data, o)
        o += 12
        if o + 4 * k > len(data):   # the last record, cut off when the engine stopped
            break
        calls.append((il, kind, set(struct.unpack_from("<%di" % k, data, o))))
        o += 4 * k
    return calls


def global_lru(trace, slots, blocks, skip=()):
    per = slots // len(blocks)
    lru = OrderedDict(((b, e), None) for b in blocks for e in range(min(per, NE)))
    miss = collections.Counter()
    for il, kind, ids in trace:
        if il in skip:
            continue
        want = {(il, e) for e in ids}
        for x in want:
            if x in lru:
                lru.move_to_end(x)
                continue
            miss[kind] += 1
            if len(lru) >= slots:
                del lru[next(y for y in lru if y not in want)]
            lru[x] = None
    return miss


def partitioned(trace, alloc):
    lrus = {b: OrderedDict(((b, e), None) for e in range(min(n, NE))) for b, n in alloc.items()}
    miss = collections.Counter()
    for il, kind, ids in trace:
        lru, n = lrus[il], alloc[il]
        want = {(il, e) for e in ids}
        for x in want:
            if x in lru:
                lru.move_to_end(x)
                continue
            miss[kind] += 1
            if len(lru) >= max(n, len(want)):
                del lru[next(y for y in lru if y not in want)]
            lru[x] = None
        while len(lru) > n:
            del lru[next(iter(lru))]
    return miss


def sized_from(train, slots, blocks, step=16, floor=16):
    def misses(b, n):
        m = partitioned([c for c in train if c[0] == b], {b: n})
        return m[0] + m[1]
    sizes = sorted(set(list(range(floor, NE + 1, step)) + [NE]))
    curve = {b: {n: misses(b, n) for n in sizes} for b in blocks}
    alloc = {b: floor for b in blocks}
    left = slots - floor * len(blocks)
    while left >= step:
        best = max(blocks, key=lambda b: curve[b][alloc[b]] - curve[b][alloc[b] + step] if alloc[b] + step <= NE else -1)
        if alloc[best] + step > NE:
            break
        alloc[best] += step
        left -= step
    return alloc


def belady(trace, slots, blocks):
    seq = [(kind, {(il, e) for e in ids}) for il, kind, ids in trace]
    nxt, next_use, first = {}, [None] * len(seq), {}
    for i in range(len(seq) - 1, -1, -1):
        next_use[i] = {x: nxt.get(x, float("inf")) for x in seq[i][1]}
        for x in seq[i][1]:
            nxt[x] = i
    for i, (_, want) in enumerate(seq):
        for x in want:
            first.setdefault(x, i)
    per = slots // len(blocks)
    res, heap = {}, []
    for b in blocks:
        for e in range(min(per, NE)):
            x = (b, e)
            res[x] = first.get(x, float("inf"))
            heapq.heappush(heap, (-res[x], x))
    miss = collections.Counter()
    for i, (kind, want) in enumerate(seq):
        for x in want:
            if x not in res:
                miss[kind] += 1
                if len(res) >= slots:
                    kept = []
                    while True:
                        nu, y = heapq.heappop(heap)
                        if y not in res or res[y] != -nu:
                            continue
                        if y in want:
                            kept.append((nu, y))
                            continue
                        del res[y]
                        break
                    for k in kept:
                        heapq.heappush(heap, k)
            res[x] = next_use[i][x]
            heapq.heappush(heap, (-res[x], x))
    return miss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--profile", help="another trace to size the per-block partitions from")
    ap.add_argument("--slots", type=int, default=4874)
    a = ap.parse_args()
    test = read(a.trace)
    train = read(a.profile) if a.profile else None
    blocks = sorted({il for il, _, _ in test} | ({il for il, _, _ in train} if train else set()))
    look = collections.Counter()
    for _, kind, ids in test:
        look[kind] += len(ids)
    print("%s: %d calls (decode %d lookups, prompt rounds %d), blocks %d..%d, %d slots" % (
        a.trace, len(test), look[0], look[1], blocks[0], blocks[-1], a.slots))
    rows = [("one LRU, an equal fill (the cache)", global_lru(test, a.slots, blocks)),
            ("the same, no MTP block over prompts", global_lru(test, a.slots, blocks, skip=(blocks[-1],))),
            ("per-block LRU, equal shares", partitioned(test, {b: a.slots // len(blocks) for b in blocks}))]
    if train:
        rows.append(("per-block LRU, sized from --profile", partitioned(test, sized_from(train, a.slots, blocks))))
    rows.append(("Belady's optimum (the bound)", belady(test, a.slots, blocks)))
    for name, m in rows:
        print("  %-38s decode %5.1f%%  prompt %5.1f%%  all %5.1f%%" % (
            name, 100.0 * m[0] / max(1, look[0]), 100.0 * m[1] / max(1, look[1]),
            100.0 * (m[0] + m[1]) / max(1, look[0] + look[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
