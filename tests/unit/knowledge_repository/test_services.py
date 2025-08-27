"""Unit tests for knowledge repository services."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from knowledge_repository.services import KnowledgeRepositoryService
from knowledge_repository.models import KnowledgeEntry

import tempfile

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
    leadership_entries = service.list_entries(entry_type="leadership")
    assert len(all_entries) == 2
    assert len(leadership_entries) == 1
    assert leadership_entries[0].id == "1"
    os.remove(db_path)
