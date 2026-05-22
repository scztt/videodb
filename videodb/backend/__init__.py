"""Read-side abstraction over analysis storage.

Today there's one backend (`SidecarBackend`) that reads `.videodb.json`
files from disk. The protocol is shaped so a future `ServerBackend`
(HTTP/OSC to an indexed server) can drop in without changing the CLI.
"""

from __future__ import annotations

from .base import Backend, FileAnalyses
from .sidecar_backend import SidecarBackend

__all__ = ["Backend", "FileAnalyses", "SidecarBackend"]
