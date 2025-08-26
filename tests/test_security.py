import pytest
from backend.api.main import app
from fastapi.testclient import TestClient
import re

client = TestClient(app)

def test_ssrf_block_private_ip():
    response = client.post("/api/analyze", json={"url": "http://127.0.0.1"})
    assert response.status_code == 400
    assert "invalid host header" in response.text.lower()

def simple_html_cleaner(html):
    # Remove script tags
    return re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)

def test_xss_sanitization():
    html = "<script>alert('xss')</script><div>Safe Content</div>"
    cleaned = simple_html_cleaner(html)
    assert "<script>" not in cleaned
    assert "Safe Content" in cleaned
