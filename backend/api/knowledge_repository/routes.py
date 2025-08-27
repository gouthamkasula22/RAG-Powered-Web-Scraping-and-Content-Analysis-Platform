"""API routes for knowledge repository."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add the project root to Python path to access src modules
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.knowledge_repository.services import KnowledgeRepositoryService
    from src.knowledge_repository.models import KnowledgeEntry
except ImportError as e:
    print(f"Import error in routes: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

router = APIRouter()
service = KnowledgeRepositoryService()

class KnowledgeEntryRequest(BaseModel):
	id: str
	type: str
	data: dict

@router.post("/knowledge-entry", response_model=KnowledgeEntry)
def create_entry(entry: KnowledgeEntryRequest):
	knowledge_entry = KnowledgeEntry(**entry.dict())
	service.add_entry(knowledge_entry)
	return knowledge_entry

@router.get("/knowledge-entry/{entry_id}", response_model=KnowledgeEntry)
def get_entry(entry_id: str):
	entry = service.get_entry(entry_id)
	if not entry:
		raise HTTPException(status_code=404, detail="Entry not found")
	return entry

@router.get("/knowledge-entries", response_model=list[KnowledgeEntry])
def list_entries(entry_type: str = None):
	return service.list_entries(entry_type=entry_type)
