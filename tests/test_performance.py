import pytest
import time
from backend.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_scraping_performance():
    start = time.time()
    response = client.post("/api/analyze", json={"url": "https://www.bbc.com"})
    duration = time.time() - start
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        assert duration < 30

def test_bulk_analysis_error_rate():
    urls = ["https://www.bbc.com", "https://www.apple.com"]
    response = client.post("/api/v1/analyze/bulk", json={"urls": urls})
    if response.status_code == 200:
        results = response.json()["results"]
        error_count = sum(1 for r in results if r["status"] == "failed")
        assert error_count / len(results) < 0.5  # Allow up to 50% for network issues
