"""Minimal content analysis pipeline test (sync wrapper)."""
import os
import asyncio
from pathlib import Path
import sys
import pytest

project_root = Path(__file__).parent.parent
backend_root = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_root))

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from backend.src.infrastructure.llm.service import ProductionLLMService, LLMServiceConfig
from backend.src.application.services.content_analysis import ContentAnalysisService
from backend.src.domain.models import AnalysisType


@pytest.mark.integration
@pytest.mark.skipif(not (os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")), reason="No LLM API keys configured")
def test_content_analysis_pipeline():
    async def _run():
        config = LLMServiceConfig()
        llm_service = ProductionLLMService(config)
        health = llm_service.get_health_status()
        print(f"Health: {health['healthy']}")

        class MockScrapingProxy:
            async def secure_scrape(self, url):
                from backend.src.domain.models import (ScrapingResult, URLInfo, ContentMetrics, ScrapedContent, ContentType, ScrapingStatus)
                from datetime import datetime
                url_info = URLInfo.from_url(url)
                text = "Mock pipeline test content." * 2
                headings = ["H1", "H2"]
                links = ["https://example.com"]
                metrics = ContentMetrics.calculate(content=text, links=links, headings=headings)
                content = ScrapedContent(
                    url_info=url_info,
                    title="Mock",
                    headings=headings,
                    main_content=text,
                    links=links,
                    meta_description="desc",
                    meta_keywords=["mock"],
                    content_type=ContentType.ARTICLE,
                    metrics=metrics,
                    scraped_at=datetime.now(),
                    status=ScrapingStatus.SUCCESS
                )
                return ScrapingResult(request=None, content=content, status_code=200, response_time=0.05, is_success=True)

        service = ContentAnalysisService(MockScrapingProxy(), llm_service)
        result = await service.analyze_content(
            content="Short pipeline test content about a product.",
            url="https://example.com/test",
            analysis_type=AnalysisType.COMPREHENSIVE
        )
        assert result is not None
        status_val = result.status.value if hasattr(result.status, 'value') else result.status
        print(f"Status: {status_val}")
        if status_val == 'completed':
            assert result.metrics is not None
        else:
            print(f"Error: {result.error_message}")

    asyncio.run(_run())


if __name__ == "__main__":
    test_content_analysis_pipeline()
