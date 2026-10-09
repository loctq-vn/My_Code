# 🎬 Video Retrieval UI

Subtitle + multi-modal **hybrid search** over lecture / news videos with a
Streamlit frontend. Retrieves the exact **shot** (timestamp range) matching
your query, then plays the video from that timestamp.

## ✨ Features

- **Hybrid retrieval** — vector (BAAI/bge-m3, 1024-d) **+** BM25 keyword over
  subtitles/captions, fused with **Reciprocal Rank Fusion (RRF)**.
- **Visual scoring (SigLIP)** - cross-modal keyframe re-ranking
  (`google/siglip-so400m-patch14-384`, 1152-d): Hybrid + Visual Re-rank
  with a tunable text/visual weight slider, plus Visual-Only search by
  image upload or frame description.
- **LanceDB** single embedded store (vector ANN + Tantivy full-text), with an
  optional **FAISS** backend behind the same interface.
- **In-memory metadata** — 2,883 shot chunks across 184 videos (L25 + L30)
  loaded once into RAM; O(1) lookups by `(video_id, shot_id)`.
- **Streamlit UI** — sidebar controls (mode / top-k / dataset **and video**
  filters), keyframe thumbnails, timestamp search, **timestamp playback**
  (local mp4 via `st.video(start_time=…)` or YouTube fallback), and a one-click
  **CSV download** that samples up to 100 `(video_name, frame_id)` rows evenly
  across the current result shots.

## 🗂 Project layout

```
my-video-retrieval/
├── app.py                     # Streamlit entrypoint (composition root)
├── config/settings.py         # Central typed config + env overrides (VRI_*)
├── core/
│   ├── schemas.py             # ShotChunk / VideoMeta / SearchResult / SearchQuery
│   └── metadata_store.py      # In-memory registry (captions + titles + videos)
├── ingestion/
│   ├── embedder.py            # bge-m3 wrapper (query encode + load .npy)
│   └── build_index.py         # CLI: captions + .npy -> LanceDB + FTS index
├── retrieval/
│   ├── base.py                # BaseRetriever + RawHit contract
│   ├── lancedb_retriever.py   # vector / BM25 / hybrid-RRF
|   |-- visual_scorer.py       # SigLIP keyframe scoring + fusion helpers
│   └── search_service.py      # query -> enriched SearchResult
├── ui/
│   ├── components.py          # Streamlit rendering
│   └── player.py              # playback spec (local / YouTube)
├── scripts/smoke_test_search.py
|   |-- smoke_test_visual.py     # visual-modes headless smoke
└── tests/                     # pytest (metadata_store, embedder)
```

## 🚀 Quickstart

```bash
# 0) (one-time) create + activate a venv with GPU torch if desired
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -r ../requirements.txt    # from My_Code root

# 1) build the index (captions + precomputed caption_embeddings/*.npy)
python -m ingestion.build_index --rebuild

# 2) headless smoke test (no UI)
python -m scripts.smoke_test_search

# 2b) visual modes smoke (downloads SigLIP ~2.8 GB once)
python -m scripts.smoke_test_visual

# 3) launch the UI
streamlit run app.py
```

> The index only needs rebuilding when the dataset/captions change
> (`--rebuild` overwrites). First run downloads bge-m3 from HuggingFace Hub.

## ⚙️ Configuration

Defaults live in `config/settings.py` and can be overridden via an app-level
`my-video-retrieval/.env` or `VRI_*` environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `VRI_DATASET_GLOB` | `*_data` | discover dataset folders (L25_data, L30_data, …) |
| `VRI_EMBED_MODEL` | `BAAI/bge-m3` | encoder — must match the stored vectors |
| `VRI_EMBED_DIM` | `1024` | vector dimension (must match `.npy`) |
| `VRI_LANCEDB_URI` | `data/index` | LanceDB storage dir |
| `VRI_TABLE_NAME` | `shots` | LanceDB table name |
| `VRI_DEFAULT_TOP_K` | `10` | results per query |
| `VRI_MAX_TOP_K` | `50` | hard cap enforced by `SearchService` |
| `VRI_RRF_K` | `60` | RRF constant used in hybrid fusion |
| `VRI_FTS_TOKENIZER` | `simple` | BM25 tokenizer (ascii_folding on by default) |
| `VRI_SIGLIP_MODEL` | `google/siglip-so400m-patch14-384` | cross-modal encoder for keyframe scoring |
| `VRI_SIGLIP_DIM` | `1152` | stored keyframe embedding dimension |
| `VRI_KEYFRAME_EMBEDDINGS_SUBDIR` | `keyframe_embeddings` | per-video SigLIP vectors |

## 🧠 Architecture & data flow

```
SearchQuery ─▶ SearchService
                 │ normalize (cap top_k, default HYBRID)
                 ▼
             LanceDBRetriever
                 ├─ VECTOR : bge-m3 encode → ANN → 1/(1+distance)
                 ├─ KEYWORD: Tantivy BM25 (subtitle+caption, ascii_folding)
                 |- HYBRID      : vector U BM25 -> RRF (sum 1/(k+rank))
                 |- VISUAL      : SigLIP keyframes (re-rank w / global)
                 │
                 ▼ enrich via MetadataStore (ShotChunk + VideoMeta + snippet)
             → [SearchResult] → Streamlit UI (cards + timestamp playback)
```

**Consistency contract (embedding):** the query encoder must match the offline
pipeline from `src/making_caption_embedding.ipynb` — same `BAAI/bge-m3`, text =
`caption` else `subtitle`, `normalize_embeddings=True`, cast to `float32`, no
instruction prefix. `Embedder.load_shot_embeddings()` prefers the on-disk
`caption_embeddings/*.npy` and falls back to re-encoding if shapes mismatch.

## Search modes

Modes in the sidebar:

| Mode | Ranking |
|---|---|
| Hybrid | BM25 + bge-m3 vector fused with RRF (pure text). |
| Hybrid + Visual Re-rank | Expands the hybrid pool (~3x top-k), scores each shot's keyframes vs the query with SigLIP (max-pooled per shot), then blends `S_final = w*S_rrf_norm + (1-w)*S_vis_norm` (both min-max scaled). Slider `w`, default 0.5. Shots without a usable keyframe keep their text position unchanged. |
| Visual-Only | Encodes an uploaded image OR the text description with SigLIP and ranks every keyframe globally (max-pooled per shot). |
| Caption Embedding only | bge-m3 vector search over the caption-embedding space only (no lexical, no subtitle). |
| Subtitle raw text only | Lexical BM25 over the raw `subtitle` text only (in-memory, no caption, no embeddings). Handy when you want the literal spoken words to drive ranking. |

Notes:

- SigLIP loads lazily on the first visual query (~2.8 GB download, once).
- Keyframe embeddings live in a different space than captions (1152-d vs
  1024-d); the two are never compared directly.
- Layout contract: `keyframes/<video>/001.jpg..N.jpg` maps 1:1 in natural
  order to the rows of `keyframe_embeddings/<video>.npy`.

## ⚠️ Notes & known data limitations

- Some caption JSONs already contain `?` where Vietnamese diacritics should be
  (e.g. `L30_V029`) — this is a source-data artifact, reproduced verbatim in
  the index. The pipeline handles UTF-8 correctly.
- Windows console: run scripts with `PYTHONIOENCODING=utf-8` (the smoke script
  forces UTF-8 itself) to print Vietnamese correctly.
- LanceDB's built-in `query_type="hybrid"` requires a registered embedding
  function; this app does its own RRF fusion instead for full control.

## ✅ Testing

```bash
python -m pytest -q          # 29 tests (metadata, embedder, visual, fusion)
```