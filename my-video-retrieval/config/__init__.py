"""Configuration package for the Video Retrieval UI.

Central, typed settings for paths, the embedding model, the LanceDB index,
and search defaults. See :mod:`config.settings`.
"""

from .settings import (
    PACKAGE_ROOT,
    SETTINGS,
    Settings,
    WORKSPACE_ROOT,
)

__all__ = [
    "SETTINGS",
    "Settings",
    "PACKAGE_ROOT",
    "WORKSPACE_ROOT",
]