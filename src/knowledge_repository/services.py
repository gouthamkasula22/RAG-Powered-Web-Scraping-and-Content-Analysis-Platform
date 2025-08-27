"""Business logic for knowledge repository."""

from typing import List, Optional
from .models import KnowledgeEntry
import sqlite3
import json

class KnowledgeRepositoryService:
    """Service for managing knowledge repository entries with SQLite persistence."""
    def __init__(self, db_path: str = "data/knowledge_repository.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add_entry(self, entry: KnowledgeEntry) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "REPLACE INTO knowledge_entries (id, type, data) VALUES (?, ?, ?)",
            (entry.id, entry.type, json.dumps(entry.data))
        )
        conn.commit()
        conn.close()

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, type, data FROM knowledge_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return KnowledgeEntry(id=row[0], type=row[1], data=json.loads(row[2]))
        return None

    def list_entries(self, entry_type: Optional[str] = None) -> List[KnowledgeEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if entry_type:
            cursor.execute("SELECT id, type, data FROM knowledge_entries WHERE type = ?", (entry_type,))
        else:
            cursor.execute("SELECT id, type, data FROM knowledge_entries")
        rows = cursor.fetchall()
        conn.close()
        return [KnowledgeEntry(id=row[0], type=row[1], data=json.loads(row[2])) for row in rows]
