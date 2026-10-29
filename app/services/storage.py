from datetime import timedelta
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def delete(self, key: str) -> None: ...

    def upload_url(self, key: str, content_type: str, max_bytes: int, ttl_s: int) -> tuple[str, dict[str, str]]:
        """A URL the browser can PUT one file to directly, plus the headers it must send."""
        ...


class LocalStorage:
    # Development only: uploads go to the app's own /dev/uploads route.
    def __init__(self, root: str):
        self.root = Path(root)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.is_file() else None

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)

    def upload_url(self, key: str, content_type: str, max_bytes: int, ttl_s: int) -> tuple[str, dict[str, str]]:
        return f"/dev/{key}", {"Content-Type": content_type}


class GcsStorage:
    def __init__(self, bucket: str):
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.bucket.blob(key).upload_from_string(data, content_type=content_type)

    def get(self, key: str) -> bytes | None:
        blob = self.bucket.blob(key)
        return blob.download_as_bytes() if blob.exists() else None

    def delete(self, key: str) -> None:
        from google.api_core.exceptions import NotFound

        try:
            self.bucket.blob(key).delete()
        except NotFound:
            pass

    def upload_url(self, key: str, content_type: str, max_bytes: int, ttl_s: int) -> tuple[str, dict[str, str]]:
        """V4 signed PUT URL. On Cloud Run there is no key file, so signing goes through
        the IAM signBlob API with the runtime service account's own token."""
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        headers = {"Content-Type": content_type, "x-goog-content-length-range": f"0,{max_bytes}"}
        url = self.bucket.blob(key).generate_signed_url(
            version="v4",
            method="PUT",
            expiration=timedelta(seconds=ttl_s),
            content_type=content_type,
            headers={"x-goog-content-length-range": headers["x-goog-content-length-range"]},
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )
        return url, headers
