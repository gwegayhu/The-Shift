from tinydb import TinyDB, Query
from typing import Dict, Any, List

class NoSQLManager:
    """Manages dynamic, unstructured metadata payloads using a JSON document store."""
    def __init__(self, db_path: str = "data/metadata.json"):
        self.db = TinyDB(db_path)

    def log_clickstream(self, order_id: int, dynamic_payload: Dict[str, Any]):
        """Stores schema-less payloads (device signatures, click maps, nested JSON)."""
        dynamic_payload["order_id"] = order_id
        self.db.insert(dynamic_payload)

    def get_logs_by_order(self, order_id: int) -> List[Dict[str, Any]]:
        Log = Query()
        return self.db.search(Log.order_id == order_id)
