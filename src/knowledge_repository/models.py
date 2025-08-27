"""Data models for knowledge repository entries."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KnowledgeEntry(BaseModel):
    """Generic entry for the knowledge repository."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    type: str = Field(..., description="Type of entry, e.g., 'leadership', 'project', 'faq'")
    data: Dict[str, Any] = Field(default_factory=dict, description="Flexible data payload")

    class Config:
        orm_mode = True
