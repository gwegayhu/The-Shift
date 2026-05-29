import sqlite3
from typing import Dict, Any, List

class SQLManager:
    """Manages transactional relational data using local SQLite."""
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_order(self, user_id: str, amount: float, status: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO orders (user_id, amount, status) VALUES (?, ?, ?)",
                (user_id, amount, status)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Executes relational aggregations (Must-Know SQL concept)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), SUM(amount) FROM orders WHERE user_id = ?", (user_id,)
            )
            count, total = cursor.fetchone()
            return {"total_orders": count or 0, "total_spent": total or 0.0}
