"""Thin async wrapper over S3/MinIO for storing and retrieving session media.

Layout inside the bucket:
  sessions/<session_id>/manifest.json     session metadata + media keys
  sessions/<session_id>/frames/<n>.jpg    captured video frames
  sessions/<session_id>/audio.webm         recorded audio track
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import get_settings


class ObjectStore:
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, bucket: str
    ) -> None:
        self._bucket = bucket
        self._session = aioboto3.Session()
        self._client_kwargs = {
            "service_name": "s3",
            "endpoint_url": endpoint,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "config": Config(signature_version="s3v4"),
        }

    def _client(self):  # type: ignore[no-untyped-def]
        return self._session.client(**self._client_kwargs)

    async def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist — safe to call on every startup."""
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket)
            except ClientError:
                await s3.create_bucket(Bucket=self._bucket)

    async def put(self, key: str, body: bytes, content_type: str) -> None:
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self._bucket, Key=key, Body=body, ContentType=content_type
            )

    async def get(self, key: str) -> tuple[bytes, str]:
        """Return (body, content_type). Raises KeyError if the object is missing."""
        async with self._client() as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except ClientError as error:
                raise KeyError(key) from error
            body = await response["Body"].read()
            return body, response.get("ContentType", "application/octet-stream")

    async def stream(self, key: str, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        async with self._client() as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except ClientError as error:
                raise KeyError(key) from error
            async for chunk in response["Body"].iter_chunks(chunk_size):
                yield chunk

    async def list_prefixes(self, prefix: str) -> list[str]:
        """Return the immediate 'subdirectory' names under a prefix (delimited by '/')."""
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            common_prefixes: list[str] = []
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=prefix, Delimiter="/"
            ):
                for entry in page.get("CommonPrefixes", []):
                    common_prefixes.append(entry["Prefix"])
            return common_prefixes

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a prefix. Returns the number of objects removed."""
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            deleted = 0
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if keys:
                    await s3.delete_objects(Bucket=self._bucket, Delete={"Objects": keys})
                    deleted += len(keys)
            return deleted


@lru_cache
def get_object_store() -> ObjectStore:
    settings = get_settings()
    return ObjectStore(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
    )
