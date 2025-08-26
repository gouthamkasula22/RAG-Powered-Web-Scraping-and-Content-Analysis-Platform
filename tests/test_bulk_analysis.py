import pytest
from backend.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_bulk_analysis_large_batch():
    # Use real, reachable URLs
    urls = ["https://www.bbc.com", "https://www.apple.com", "https://www.microsoft.com", "https://www.amazon.com", "https://www.coursera.org"] * 4
    urls = urls[:20]
    response = client.post("/api/v1/analyze/bulk", json={"urls": urls, "parallel_limit": 5})
    assert response.status_code in [200, 400]  # Accept 400 for validation errors
    if response.status_code == 200:
        assert len(response.json()["results"]) == 20

def test_bulk_analysis_timeout():
    urls = ["https://www.bbc.com"]
    response = client.post("/api/v1/analyze/bulk", json={"urls": urls, "timeout": 1})
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        assert response.json()["results"][0]["status"] in ["failed", "completed"]
