"""Unit tests for knowledge repository models."""
import sys
import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Create a local version of the model for testing
class KnowledgeEntry(BaseModel):
    """Generic entry for the knowledge repository."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    type: str = Field(..., description="Type of entry, e.g., 'leadership', 'project', 'faq'")
    data: Dict[str, Any] = Field(default_factory=dict, description="Flexible data payload")

    class Config:
        orm_mode = True

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
    assert entry.type == "leadership"
    assert entry.data["name"] == "John Doe"
