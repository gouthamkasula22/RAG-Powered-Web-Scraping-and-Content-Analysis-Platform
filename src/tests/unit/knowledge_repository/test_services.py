from src.knowledge_repository.services import KnowledgeRepositoryService
from src.knowledge_repository.models import KnowledgeEntry

def test_service_add_and_get_entry():
    service = KnowledgeRepositoryService()
    entry = KnowledgeEntry(id="1", type="leadership", data={"name": "Jane Smith"})
    service.add_entry(entry)
    retrieved = service.get_entry("1")
    assert retrieved is not None
    assert retrieved.data["name"] == "Jane Smith"

def test_service_list_entries():
    service = KnowledgeRepositoryService()
    entry1 = KnowledgeEntry(id="1", type="leadership", data={"name": "A"})
    entry2 = KnowledgeEntry(id="2", type="project", data={"title": "B"})
    service.add_entry(entry1)
    service.add_entry(entry2)
    all_entries = service.list_entries()
    leadership_entries = service.list_entries(entry_type="leadership")
    assert len(all_entries) == 2
    assert len(leadership_entries) == 1
    assert leadership_entries[0].id == "1"
