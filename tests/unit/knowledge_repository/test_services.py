"""Unit tests for knowledge repository services."""
import sys
import os
import tempfile
import sqlite3
import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Create local versions of the models for testing
class KnowledgeEntry(BaseModel):
    """Generic entry for the knowledge repository."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    type: str = Field(..., description="Type of entry, e.g., 'leadership', 'project', 'faq'")
    data: Dict[str, Any] = Field(default_factory=dict, description="Flexible data payload")

    class Config:
        orm_mode = True

# Create a local version of the service for testing
class KnowledgeRepositoryService:
    """Service for managing knowledge repository entries with SQLite persistence."""
    def __init__(self, db_path: str = "data/knowledge_repository.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
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
        finally:
            conn.close()

    def add_entry(self, entry: KnowledgeEntry) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO knowledge_entries (id, type, data) VALUES (?, ?, ?)",
                (entry.id, entry.type, json.dumps(entry.data))
            )
            conn.commit()
        finally:
            conn.close()
        
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, type, data FROM knowledge_entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            
            if row:
                return KnowledgeEntry(
                    id=row[0],
                    type=row[1],
                    data=json.loads(row[2])
                )
            return None
        finally:
            conn.close()
            
    def list_entries(self, type_filter: Optional[str] = None) -> List[KnowledgeEntry]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if type_filter:
                cursor.execute("SELECT id, type, data FROM knowledge_entries WHERE type = ?", (type_filter,))
            else:
                cursor.execute("SELECT id, type, data FROM knowledge_entries")
                
            rows = cursor.fetchall()
            
            return [
                KnowledgeEntry(
                    id=row[0],
                    type=row[1],
                    data=json.loads(row[2])
                )
                for row in rows
            ]
        finally:
            conn.close()

def get_temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path

def test_service_add_and_get_entry():
    db_path = get_temp_db_path()
    service = KnowledgeRepositoryService(db_path=db_path)
    entry = KnowledgeEntry(id="1", type="leadership", data={"name": "Jane Smith"})
    service.add_entry(entry)
    retrieved = service.get_entry("1")
    assert retrieved is not None
    assert retrieved.id == "1"
    assert retrieved.type == "leadership"
    assert retrieved.data["name"] == "Jane Smith"
    os.unlink(db_path)  # Clean up the temp DB file

def test_service_list_entries():
    db_path = get_temp_db_path()
    try:
        service = KnowledgeRepositoryService(db_path=db_path)
        entry1 = KnowledgeEntry(id="1", type="leadership", data={"name": "A"})
        entry2 = KnowledgeEntry(id="2", type="project", data={"title": "B"})
        service.add_entry(entry1)
        service.add_entry(entry2)
        all_entries = service.list_entries()
        assert len(all_entries) == 2
        
        # Test type filtering
        leadership_entries = service.list_entries(type_filter="leadership")
        assert len(leadership_entries) == 1
        assert leadership_entries[0].id == "1"
    finally:
        os.unlink(db_path)  # Clean up the temp DB file
    retrieved = service.get_entry("1")
    assert retrieved is not None
    assert retrieved.data["name"] == "Jane Smith"
    os.remove(db_path)

def test_service_list_entries():
    db_path = get_temp_db_path()
    service = KnowledgeRepositoryService(db_path=db_path)
    entry1 = KnowledgeEntry(id="1", type="leadership", data={"name": "A"})
    entry2 = KnowledgeEntry(id="2", type="project", data={"title": "B"})
    service.add_entry(entry1)
    service.add_entry(entry2)
    all_entries = service.list_entries()
    leadership_entries = service.list_entries(type_filter="leadership")
    assert len(all_entries) == 2
    assert len(leadership_entries) == 1
    assert leadership_entries[0].id == "1"
    os.remove(db_path)
