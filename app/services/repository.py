from typing import Protocol

from app.schemas import Scan


class ScanRepository(Protocol):
    def save(self, scan: Scan) -> None: ...

    def get(self, scan_id: str) -> Scan | None: ...


class MemoryRepository:
    def __init__(self):
        self._scans: dict[str, Scan] = {}

    def save(self, scan: Scan) -> None:
        self._scans[scan.id] = scan

    def get(self, scan_id: str) -> Scan | None:
        return self._scans.get(scan_id)


class FirestoreRepository:
    def __init__(self, collection: str):
        from google.cloud import firestore

        self.collection = firestore.Client().collection(collection)

    def save(self, scan: Scan) -> None:
        self.collection.document(scan.id).set(scan.model_dump(mode="json"))

    def get(self, scan_id: str) -> Scan | None:
        doc = self.collection.document(scan_id).get()
        return Scan.model_validate(doc.to_dict()) if doc.exists else None
