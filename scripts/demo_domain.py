#!/usr/bin/env python3
"""
Demo script showcasing the domain models and services.
Run this to see our domain layer in action.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain import (
    URLInfo, ScrapedContent, ScrapingRequest, ContentMetrics,
    ContentType, ScrapingStatus,
    ContentClassificationService, ContentQualityService, URLAnalysisService,
    ValidationError, ContentQualityError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_url_info():
    """Demonstrate URLInfo functionality"""
    print("\n" + "="*60)
    print("🔗 URLInfo Demo")
    print("="*60)
    
    urls = [
        "https://example.com/",
        "https://blog.example.com/article/web-scraping-guide",
        "https://shop.example.com/product/123?color=red&size=large",
        "https://news.example.com/breaking-news-update"
    ]
    
    for url in urls:
        try:
            url_info = URLInfo.from_url(url)
            print(f"\n📍 URL: {url}")
            print(f"   Domain: {url_info.domain}")
            print(f"   Base Domain: {url_info.base_domain}")
            print(f"   Secure: {url_info.is_secure}")
            print(f"   Root Page: {url_info.is_root_page}")
            print(f"   Path: {url_info.path}")
            if url_info.query_params:
                print(f"   Query Params: {url_info.query_params}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_content_metrics():
    """Demonstrate ContentMetrics functionality"""
    print("\n" + "="*60)
    print("📊 ContentMetrics Demo")
    print("="*60)
    
    sample_contents = [
        {
            "name": "Short Blog Post",
            "content": "This is a short blog post. It has minimal content for demonstration.",
            "links": ["https://example.com/ref1"],
            "headings": ["Short Post"]
        },
        {
            "name": "Comprehensive Article", 
            "content": " ".join([
                "This is a comprehensive article about web content analysis.",
                "It contains multiple sentences and paragraphs.",
                "The article discusses various techniques for extracting meaningful information from web pages.",
                "Content analysis involves examining text structure, readability, and quality metrics."
            ] * 10),  # Repeat for substantial content
            "links": ["https://example.com/ref1", "https://example.com/ref2"],
            "headings": ["Introduction", "Methods", "Analysis", "Conclusion"]
        }
    ]
    
    for sample in sample_contents:
        print(f"\n📝 {sample['name']}")
        metrics = ContentMetrics.calculate(
            sample['content'], 
            sample['links'], 
            sample['headings']
        )
        
        print(f"   Words: {metrics.word_count}")
        print(f"   Sentences: {metrics.sentence_count}")
        print(f"   Paragraphs: {metrics.paragraph_count}")
        print(f"   Links: {metrics.link_count}")
        print(f"   Headings: {metrics.heading_count}")
        print(f"   Reading Time: {metrics.reading_time_minutes:.1f} min")
        print(f"   Density Score: {metrics.content_density_score:.2f}/10")


def demo_content_classification():
    """Demonstrate ContentClassificationService"""
    print("\n" + "="*60)
    print("🏷️ Content Classification Demo")
    print("="*60)
    
    test_contents = [
        {
            "url": "https://example.com/",
            "title": "Welcome to Example Corp",
            "content": "Welcome to our homepage. We offer various services and products.",
            "headings": ["About Us", "Services", "Contact"]
        },
        {
            "url": "https://news.example.com/breaking-update",
            "title": "Breaking News: Important Update",
            "content": "This is breaking news about an important development. Stay tuned for updates.",
            "headings": ["Latest Update", "Details"]
        },
        {
            "url": "https://blog.example.com/post/web-scraping-guide",
            "title": "Complete Guide to Web Scraping",
            "content": "This blog post covers everything you need to know about web scraping techniques and best practices.",
            "headings": ["Introduction", "Getting Started", "Advanced Topics"]
        },
        {
            "url": "https://shop.example.com/product/laptop-xyz",
            "title": "Premium Laptop - Buy Now",
            "content": "This premium laptop offers excellent performance. Price: $999. Add to cart for fast shipping.",
            "headings": ["Specifications", "Reviews", "Purchase Options"]
        }
    ]
    
    for test_content in test_contents:
        url_info = URLInfo.from_url(test_content["url"])
        metrics = ContentMetrics.calculate(test_content["content"], [], test_content["headings"])
        
        # Create ScrapedContent for classification
        content = ScrapedContent(
            url_info=url_info,
            title=test_content["title"],
            headings=test_content["headings"],
            main_content=test_content["content"] * 10,  # Make it long enough
            links=[],
            meta_description=None,
            meta_keywords=[],
            content_type=ContentType.UNKNOWN,  # Will be classified
            metrics=ContentMetrics.calculate(test_content["content"] * 10, [], test_content["headings"]),
            scraped_at=datetime.now(),
            status=ScrapingStatus.SUCCESS
        )
        
        classification = ContentClassificationService.classify_content(content)
        
        print(f"\n🔍 URL: {test_content['url']}")
        print(f"   Title: {test_content['title']}")
        print(f"   Classification: {classification.value}")


def demo_url_analysis():
    """Demonstrate URLAnalysisService"""
    print("\n" + "="*60)
    print("🔍 URL Analysis Demo")  
    print("="*60)
    
    test_urls = [
        "https://example.com/article/interesting-topic",
        "https://api.example.com/data.json",
        "https://user.github.io/blog/post-title",
        "https://twitter.com/user/status/123456",
        "https://docs.example.com/api/reference",
    ]
    
    for url in test_urls:
        url_info = URLInfo.from_url(url)
        
        is_content = URLAnalysisService.is_likely_content_page(url_info)
        complexity = URLAnalysisService.estimate_scraping_complexity(url_info)
        timeout = URLAnalysisService.suggest_timeout(url_info)
        
        print(f"\n🌐 URL: {url}")
        print(f"   Content Page: {'✅ Yes' if is_content else '❌ No'}")
        print(f"   Complexity: {complexity}")
        print(f"   Suggested Timeout: {timeout}s")


def demo_quality_service():
    """Demonstrate ContentQualityService"""
    print("\n" + "="*60)
    print("⭐ Content Quality Demo")
    print("="*60)
    
    # Create sample content with different quality levels
    quality_samples = [
        {
            "name": "High Quality Article",
            "content": " ".join([
                "This is a high-quality article with substantial content.",
                "It provides comprehensive coverage of the topic with detailed analysis.",
                "The article is well-structured with clear headings and good readability.",
                "It contains valuable information for readers interested in the subject."
            ] * 25),  # 400+ words
            "headings": ["Introduction", "Main Analysis", "Key Points", "Conclusion"],
            "links": []  # No links to avoid density issues
        },
        {
            "name": "Low Quality Content",
            "content": "This is very short content with minimal information and poor structure.",
            "headings": [],
            "links": ["link1", "link2", "link3", "link4", "link5"]  # Too many links
        }
    ]
    
    for sample in quality_samples:
        url_info = URLInfo.from_url("https://example.com/test")
        metrics = ContentMetrics.calculate(sample["content"], sample["links"], sample["headings"])
        
        try:
            content = ScrapedContent(
                url_info=url_info,
                title=sample["name"],
                headings=sample["headings"],
                main_content=sample["content"],
                links=sample["links"],
                meta_description=None,
                meta_keywords=[],
                content_type=ContentType.ARTICLE,
                metrics=metrics,
                scraped_at=datetime.now(),
                status=ScrapingStatus.SUCCESS
            )
            
            # Calculate quality score
            score = ContentQualityService.calculate_quality_score(content)
            
            print(f"\n📄 {sample['name']}")
            print(f"   Words: {metrics.word_count}")
            print(f"   Quality Score: {score:.1f}/10")
            
            # Test quality validation
            try:
                ContentQualityService.validate_content_quality(content)
                print(f"   Quality Check: ✅ Passed")
            except ContentQualityError as e:
                print(f"   Quality Check: ❌ Failed - {e.message}")
                
        except ValueError as e:
            print(f"\n📄 {sample['name']}")
            print(f"   ❌ Content Creation Failed: {e}")


def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n" + "="*60)
    print("🛡️ Error Handling Demo")
    print("="*60)
    
    # Test various error scenarios
    error_tests = [
        {
            "name": "Invalid URL",
            "test": lambda: URLInfo.from_url("not-a-url")
        },
        {
            "name": "Invalid Timeout",
            "test": lambda: ScrapingRequest("https://example.com", timeout_seconds=-1)
        }
    ]
    
    for test in error_tests:
        print(f"\n🧪 Testing: {test['name']}")
        try:
            test["test"]()
            print("   ✅ No error (unexpected)")
        except Exception as e:
            print(f"   ❌ Error caught: {type(e).__name__}: {e}")


def main():
    """Run all demos"""
    print("🚀 Web Content Analyzer - Domain Layer Demo")
    print("=" * 80)
    print("This demo showcases the domain models and services we've implemented.")
    print("Following SOLID principles with comprehensive error handling and logging.")
    
    try:
        demo_url_info()
        demo_content_metrics()
        demo_content_classification()
        demo_url_analysis()
        demo_quality_service()
        demo_error_handling()
        
        print("\n" + "="*80)
        print("✅ Demo completed successfully!")
        print("🎯 Domain layer is working perfectly and ready for integration.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
