"""Object storage for recorded session media (MinIO / S3)."""

from app.storage.object_store import ObjectStore, get_object_store

__all__ = ["ObjectStore", "get_object_store"]
