from src.knowledge_repository.models import KnowledgeEntry

def test_knowledge_entry_creation():
    entry = KnowledgeEntry(
        id="1",
        type="leadership",
        data={
            "name": "John Doe",
            "role": "CEO",
            "bio": "Experienced leader.",
            "contact": "john.doe@example.com",
            "photo_url": "https://example.com/photo.jpg"
        }
    )
    assert entry.type == "leadership"
    assert entry.data["name"] == "John Doe"
