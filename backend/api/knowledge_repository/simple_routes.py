"""Simplified API routes for knowledge repository."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

router = APIRouter()

# In-memory storage for demo purposes (replace with database later)
knowledge_entries: Dict[str, Dict[str, Any]] = {}
entry_counter = 0

class KnowledgeEntryRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeEntryResponse(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/knowledge-entry", response_model=KnowledgeEntryResponse)
async def create_knowledge_entry(entry: KnowledgeEntryRequest):
    """Create a new knowledge entry"""
    global entry_counter
    entry_counter += 1
    entry_id = f"entry_{entry_counter}"
    
    knowledge_entries[entry_id] = {
        "id": entry_id,
        "title": entry.title,
        "content": entry.content,
        "metadata": entry.metadata or {}
    }
    
    return KnowledgeEntryResponse(**knowledge_entries[entry_id])

@router.get("/knowledge-entry/{entry_id}", response_model=KnowledgeEntryResponse)
async def get_knowledge_entry(entry_id: str):
    """Get a specific knowledge entry by ID"""
    if entry_id not in knowledge_entries:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    return KnowledgeEntryResponse(**knowledge_entries[entry_id])

@router.get("/knowledge-entries", response_model=List[KnowledgeEntryResponse])
async def list_knowledge_entries():
    """List all knowledge entries"""
    return [KnowledgeEntryResponse(**entry) for entry in knowledge_entries.values()]

@router.delete("/knowledge-entry/{entry_id}")
async def delete_knowledge_entry(entry_id: str):
    """Delete a knowledge entry"""
    if entry_id not in knowledge_entries:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    del knowledge_entries[entry_id]
    return {"message": "Entry deleted successfully"}
