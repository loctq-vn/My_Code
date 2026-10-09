"""Semantic shot segmentation using subtitle topic boundaries (bge-m3).

Reads ASR subtitle segments, embeds each with bge-m3, detects topic boundaries
from consecutive cosine similarity, then partitions the timeline with a small
dynamic program so that every shot duration stays within [MIN_DUR, MAX_DUR]
while placing as many cuts as possible on semantic boundaries. Coverage is
extended to the real video end (credits / silent tail) when media-info has it.

Output mirrors the existing shots JSON schema:
    [{"shot_id":.., "start_time":.., "end_time":.., "duration":..}]
"""
from __future__ import annotations
import bisect
import json
import math
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP_PKG = ROOT / "my-video-retrieval"
if str(APP_PKG) not in sys.path:
    sys.path.insert(0, str(APP_PKG))

from ingestion.embedder import Embedder  # noqa: E402

# ---------------- config ----------------
DATASET      = ROOT / "L25_data"
SUBTITLE_DIR = DATASET / "subtitles"
OUTPUT_DIR   = DATASET / "shots_new"
VIDEOS       = ["L25_V001", "L25_V002", "L25_V003"]
EMBED_MODEL  = "BAAI/bge-m3"
COSINE_THETA = 0.60   # absolute cut threshold (tuned down over-cutting)
STD_K        = 0.8    # relative cut: mean - k*std (floor at COSINE_THETA)
MIN_DUR      = 15.0
MAX_DUR      = 60.0
EXTEND_TO_VIDEO_END = True  # fold credits/silent tail into extra trailing shots


def load_segments(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = [{**s, "start": float(s["start"]), "end": float(s["end"])} for s in data]
    segs.sort(key=lambda s: s["start"])
    return segs


def sims_of(embedder, segments) -> np.ndarray:
    texts = [s["text"] for s in segments]
    vecs = embedder.encode_texts(texts)
    return np.array([float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)])


def cuts_from(sims: np.ndarray) -> List[int]:
    if sims.size == 0:
        return []
    mean, std = float(sims.mean()), float(sims.std())
    thresh = max(mean - STD_K * std, COSINE_THETA)
    return [i + 1 for i, s in enumerate(sims) if s < thresh]


def boundaries_of(segs: List[dict]) -> List[float]:
    """Chained boundary times: C[0]=0.0, C[i]=end of segment i-1 (no gaps)."""
    return [0.0] + [s["end"] for s in segs]


def partition_dp(C: List[float], sem: set, min_dur: float, max_dur: float) -> Optional[List[tuple]]:
    """Partition the chained span [0, C[-1]] into consecutive groups (i..j-1)
    whose duration C[j]-C[i] stays within [min_dur, max_dur]. Cost of a group
    end is -1 on a semantic boundary and +1 otherwise, i.e. the DP maximises
    (#semantic cuts - #non-semantic cuts). Returns [(i, j), ...]."""
    m = len(C) - 1
    INF = float("inf")
    eps = 1e-6
    dp = [INF] * (m + 1)
    take = [-1] * (m + 1)
    dp[0] = 0.0
    mid = 0.5 * (min_dur + max_dur)
    for j in range(1, m + 1):
        i_lo = bisect.bisect_left(C, C[j] - max_dur - eps)
        i_hi = bisect.bisect_right(C, C[j] - min_dur + eps) - 1
        best = (INF, 0.0)
        arg = -1
        for i in range(max(0, i_lo), min(j, i_hi + 1)):
            if dp[i] == INF:
                continue
            key = (dp[i] + (-1.0 if j in sem else 1.0), abs((C[j] - C[i]) - mid))
            if key < best:
                best, arg = key, i
        if arg >= 0:
            dp[j], take[j] = best[0], arg
    if dp[m] == INF:
        return None
    groups, j = [], m
    while j > 0:
        groups.append((take[j], j))
        j = take[j]
    groups.reverse()
    return groups


def partition_shots(segs: List[dict], cuts: List[int]):
    """DP partition under [MIN_DUR, MAX_DUR]. If the bounds are unsatisfiable
    because a tiny leading/trailing segment is glued to a full-MAX_DUR one,
    push that segment into the head/tail margin and retry. Never fails:
    final fallback is even time chunks. Returns (shots, C, work)."""
    sem = set(cuts)
    work = list(segs)
    drop_last = True
    while work:
        C = boundaries_of(work)
        groups = partition_dp(C, sem, MIN_DUR, MAX_DUR)
        if groups is not None:
            return [dict(start=C[i], end=C[j]) for i, j in groups], C, work
        if drop_last:
            work = work[:-1]
        else:
            work = work[1:]
            sem = {b - 1 for b in sem if b - 1 >= 1}
        drop_last = not drop_last
    # extreme fallback: even chunks over the full original span
    total = segs[-1]["end"] if segs else 0.0
    n = max(1, math.ceil(total / MAX_DUR - 1e-9))
    step = total / n
    shots = [dict(start=q * step, end=total if q == n - 1 else (q + 1) * step)
             for q in range(n)]
    return shots, [0.0] + [s["end"] for s in segs], segs


def apply_tail(shots: List[dict], C: List[float],
               video_end: Optional[float]) -> List[dict]:
    """Extend coverage to the real video end (credits / silent tail)."""
    if video_end is None or not shots:
        return shots
    end = shots[-1]["end"]
    if video_end <= end + 0.5:
        shots[-1]["end"] = max(end, video_end)
        return shots
    D = video_end - end
    if D < MIN_DUR:
        # fold the tiny tail into the last shot when it still fits MAX_DUR;
        # otherwise split the last span at a segment boundary, or evenly.
        last = shots[-1]
        total = (last["end"] - last["start"]) + D
        if total <= MAX_DUR + 1e-6:
            shots[-1]["end"] = video_end
            return shots
        mid_t = 0.5 * (last["start"] + video_end)
        cands = [t for t in C
                 if MIN_DUR - 1e-6 <= t - last["start"] <= MAX_DUR + 1e-6
                 and MIN_DUR - 1e-6 <= video_end - t <= MAX_DUR + 1e-6]
        if cands:
            t = min(cands, key=lambda x: abs(x - mid_t))
            shots[-1] = dict(start=last["start"], end=t)
            shots.append(dict(start=t, end=video_end))
            return shots
        # no usable segment boundary (e.g. single long segment): split evenly
        n = max(2, math.ceil(total / MAX_DUR - 1e-9))
        step = total / n
        t0 = last["start"]
        shots.pop()
        for q in range(n):
            t1 = video_end if q == n - 1 else t0 + step
            shots.append(dict(start=t0, end=t1))
            t0 = t1
        return shots
    n = max(1, math.ceil(D / MAX_DUR - 1e-9))
    step = D / n
    t0 = end
    for q in range(n):
        t1 = video_end if q == n - 1 else t0 + step
        shots.append(dict(start=t0, end=t1))
        t0 = t1
    return shots


def video_duration(video: str) -> Optional[float]:
    p = DATASET / "media-info" / f"{video}.json"
    if not p.is_file():
        return None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    d = info.get("duration", info.get("length"))
    try:
        return float(d) if d else None
    except (TypeError, ValueError):
        return None


def attach_texts(rows: List[dict], segs: List[dict]) -> List[dict]:
    """Attach subtitle text to each published shot (row): a segment's text
    belongs to the shot containing its midpoint (bisect over row start_times).
    Uses the rounded boundaries actually written to JSON, so `text` always
    matches the stored shot ranges. Segments trimmed from the DP fall into
    the head/tail shots; rows without speech keep text == ""."""
    starts = [r["start_time"] for r in rows]
    buckets: List[List[str]] = [[] for _ in rows]
    for seg in segs:
        mid = 0.5 * (float(seg["start"]) + float(seg["end"]))
        k = bisect.bisect_right(starts, mid) - 1
        if 0 <= k < len(rows):
            text = str(seg.get("text", "")).strip()
            if text:
                buckets[k].append(text)
    for r, texts in zip(rows, buckets):
        r["text"] = " ".join(texts)
    return rows


def main() -> None:
    print(f"model={EMBED_MODEL} min={MIN_DUR}s max={MAX_DUR}s extend_tail={EXTEND_TO_VIDEO_END}")
    embedder = Embedder()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for video in VIDEOS:
        p = SUBTITLE_DIR / f"{video}.json"
        if not p.is_file():
            print(f"SKIP {video}: no subtitle")
            continue
        segs = load_segments(p)
        vend = video_duration(video) if EXTEND_TO_VIDEO_END else None
        if not segs:
            if not vend:
                print(f"SKIP {video}: no subtitle segments and no media duration")
                continue
            n = max(1, math.ceil(vend / MAX_DUR))
            step = vend / n
            shots = [dict(start=q * step, end=vend if q == n - 1 else (q + 1) * step)
                     for q in range(n)]
            note = "no-speech: even chunks"
        else:
            sims = sims_of(embedder, segs)
            cuts = cuts_from(sims)
            shots, C, work = partition_shots(segs, cuts)
            tail_end = vend
            if tail_end is None and len(work) < len(segs):
                tail_end = segs[-1]["end"]  # keep coverage of trimmed segments
            shots = apply_tail(shots, C, tail_end)
            note = f"{len(cuts)} semantic cuts" + (
                f", {len(segs) - len(work)} trimmed" if len(work) < len(segs) else "")
        rows = [{"shot_id": k + 1,
                 "start_time": round(s["start"], 2),
                 "end_time": round(s["end"], 2),
                 "duration": round(s["end"] - s["start"], 2)}
                for k, s in enumerate(shots)]
        rows = attach_texts(rows, segs)  # attach subtitle text (no-op w/o speech)
        out_file = OUTPUT_DIR / f"{video}.json"
        out_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        durs = [r["duration"] for r in rows]
        span = vend if vend else segs[-1]["end"]
        print(f"{video}: {len(rows)} shots | min={min(durs):.1f}s max={max(durs):.1f}s "
              f"coverage={rows[-1]['end_time']:.1f}s/{span:.1f}s ({note}) -> {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


