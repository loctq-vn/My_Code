"""Streamlit entrypoint for the Video Retrieval UI.

Composition root: builds the (cached) services, wires the UI components, and
runs the search -> results -> playback flow. Keeps every layer decoupled —
no search logic here, only orchestration.

Run from the app root::

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# --- sys.path bootstrap ------------------------------------------------------
# ``streamlit run`` executes this file directly (not as a module), so the app
# root is NOT on sys.path by default. Make ``import config / core / ...`` work
# regardless of the current working directory.
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
# -----------------------------------------------------------------------------

import streamlit as st

from config.settings import SETTINGS
from core.metadata_store import MetadataStore
from core.schemas import SearchMode, SearchQuery, SearchResult
from retrieval import SearchService
from ui.components import (
    render_empty_state,
    render_header,
    render_playback_area,
    render_results,
    render_search_bar,
    render_sidebar,
)


def _save_upload_to_temp(upload) -> Optional[str]:
    """Persist a Streamlit upload to a temp file so encoders can read it."""
    if upload is None:
        return None
    suffix = os.path.splitext(upload.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getvalue())
        return tmp.name


@st.cache_resource(show_spinner="Loading metadata + search index ...")
def _load_service() -> tuple:
    """Build the metadata store + search service exactly once per process."""
    store = MetadataStore(SETTINGS)
    service = SearchService(settings=SETTINGS, metadata=store)
    return store, service


def _selected_result(results: list[SearchResult]) -> SearchResult | None:
    """Return the result the user clicked (via session_state), if present."""
    selected_id = st.session_state.get("selected_result_id")
    if not selected_id:
        return None
    for r in results:
        if f"{r.video_id}:{r.shot_id}" == selected_id:
            return r
    return None


def main() -> None:
    render_header()
    store, service = _load_service()

    # ---- sidebar + search bar -------------------------------------------------
    settings = render_sidebar(store, st.session_state.get("last_query"))
    query_text = render_search_bar()

    # ---- search --------------------------------------------------------------
    results: list[SearchResult] = []
    image_file = settings.get("image_file")
    if not query_text and image_file is None:
        render_empty_state()
    else:
        q = SearchQuery(
            text=query_text or "",
            mode=settings["mode"],
            top_k=settings["top_k"],
            datasets=tuple(settings["datasets"]) if settings["datasets"] else None,
            video_ids=tuple(settings["video_ids"]) if settings.get("video_ids") else None,
            visual_weight=settings.get("visual_weight", 0.5),
            image_path=_save_upload_to_temp(image_file),
        )
        try:
            with st.spinner("Searching ..."):
                results = service.search(q)
            st.session_state["last_query"] = query_text
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc!r}")
            results = []

        # ---- playback + results ----------------------------------------------
        selected = _selected_result(results)
        if selected is not None:
            render_playback_area(selected)
        render_results(results, metadata_store=store)

    # footer
    st.divider()
    st.caption(
        f"Index: {SETTINGS.lancedb_uri} · model {SETTINGS.embed_model} "
        f"(dim {SETTINGS.embed_dim}) · datasets {list(store.datasets())}"
    )


if __name__ == "__main__":
    main()