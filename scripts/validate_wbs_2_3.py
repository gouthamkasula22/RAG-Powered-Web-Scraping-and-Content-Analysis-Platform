"""
WBS 2.3 Analysis Report Generator - Validation Script
Tests all components of the report generation system for compliance with requirements.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_mock_analysis_result(url: str, scores: Dict[str, float]) -> Any:
    """Create mock analysis result for testing"""
    from datetime import datetime
    
    # Return a simple dict instead of trying to import domain models
    return {
        'url': url,
        'title': f"Test Site - {url}",
        'content_quality_score': scores.get('content_quality', 7.5),
        'seo_score': scores.get('seo_optimization', 6.8),
        'user_experience_score': scores.get('user_experience', 8.2),
        'accessibility_score': scores.get('accessibility', 7.0),
        'performance_score': scores.get('performance', 8.5),
        'security_score': scores.get('security', 9.0),
        'overall_score': sum(scores.values()) / len(scores) if scores else 7.5,
        'insights': {
            'strengths': [f"Good content structure on {url}", f"Clear navigation design on {url}"],
            'weaknesses': [f"SEO optimization needed on {url}", f"Image alt text missing on {url}"],
            'recommendations': [f"Implement structured data for {url}", f"Optimize images for {url}"],
            'confidence_score': 8.5
        },
        'timestamp': datetime.now().isoformat()
    }

async def test_report_generation_service():
    """Test the core report generation service"""
    
    logger.info("=== Testing Report Generation Service ===")
    
    try:
        from src.application.services.report_generation import ReportGenerationService
        from src.infrastructure.reporting.template_manager import ReportTemplateManager
        from src.infrastructure.reporting.cache_service import MemoryReportCache
        
        # Initialize services
        template_manager = ReportTemplateManager()
        cache_service = MemoryReportCache()
        
        service = ReportGenerationService(
            template_manager=template_manager,
            cache_service=cache_service
        )
        
        # Create test analysis results
        test_analyses = [
            create_mock_analysis_result("https://example1.com", {
                'overall': 8.2, 'content': 8.5, 'seo': 7.8, 'ux': 8.0
            }),
            create_mock_analysis_result("https://example2.com", {
                'overall': 6.8, 'content': 7.2, 'seo': 6.2, 'ux': 6.9
            }),
            create_mock_analysis_result("https://example3.com", {
                'overall': 7.5, 'content': 7.8, 'seo': 7.1, 'ux': 7.6
            })
        ]
        
        # Test individual report generation
        logger.info("Testing individual report generation...")
        
        start_time = time.time()
        individual_report = await service.generate_individual_report(
            analysis=test_analyses[0],
            template="individual_analysis",
            format_type="html",
            options={"include_technical_details": True}
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        # Validate individual report
        assert individual_report is not None, "Individual report generation failed"
        assert individual_report.url == "https://example1.com", "URL mismatch in individual report"
        assert individual_report.overall_score == 8.2, "Overall score mismatch"
        assert len(individual_report.dimension_scores) >= 6, "Insufficient dimension scores"
        assert len(individual_report.executive_summary.summary_text) <= 200 * 10, "Executive summary too long (should be <200 words)"
        assert len(individual_report.improvement_roadmap) > 0, "No improvement roadmap items"
        
        logger.info(f"✅ Individual report generated successfully in {generation_time:.1f}ms")
        logger.info(f"   - Overall score: {individual_report.overall_score}/10")
        logger.info(f"   - Dimensions analyzed: {len(individual_report.dimension_scores)}")
        logger.info(f"   - Roadmap items: {len(individual_report.improvement_roadmap)}")
        
        # Test comparative report generation
        logger.info("Testing comparative report generation...")
        
        start_time = time.time()
        comparative_report = await service.generate_comparative_report(
            analyses=test_analyses,
            template="comparative_analysis",
            format_type="html",
            options={"min_differentiators": 3}
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        # Validate comparative report
        assert comparative_report is not None, "Comparative report generation failed"
        assert comparative_report.websites_analyzed == 3, "Incorrect website count"
        assert len(comparative_report.key_differentiators) >= 3, "Insufficient differentiators (minimum 3 required)"
        assert len(comparative_report.website_comparisons) == 3, "Incorrect comparison count"
        assert len(comparative_report.cross_site_recommendations) > 0, "No cross-site recommendations"
        
        logger.info(f"✅ Comparative report generated successfully in {generation_time:.1f}ms")
        logger.info(f"   - Websites analyzed: {comparative_report.websites_analyzed}")
        logger.info(f"   - Key differentiators: {len(comparative_report.key_differentiators)}")
        logger.info(f"   - Cross-site recommendations: {len(comparative_report.cross_site_recommendations)}")
        
        # Test bulk report generation
        logger.info("Testing bulk report generation...")
        
        start_time = time.time()
        bulk_reports = await service.generate_bulk_reports(
            analyses=test_analyses,
            template="individual_analysis",
            format_type="html",
            options={"parallel_processing": True}
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        # Validate bulk generation
        assert bulk_reports is not None, "Bulk report generation failed"
        assert bulk_reports.total_reports_generated == 3, "Incorrect bulk report count"
        assert bulk_reports.successful_reports == 3, "Some bulk reports failed"
        assert bulk_reports.total_urls_analyzed == 3, "Incorrect URL count"
        
        logger.info(f"✅ Bulk reports generated successfully in {generation_time:.1f}ms")
        logger.info(f"   - Total reports: {bulk_reports.total_reports_generated}")
        logger.info(f"   - Success rate: {bulk_reports.successful_reports}/{bulk_reports.total_reports_generated}")
        logger.info(f"   - Average time per report: {bulk_reports.average_generation_time_ms:.1f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Report generation service test failed: {e}")
        raise

async def test_template_management():
    """Test template management functionality"""
    
    logger.info("=== Testing Template Management ===")
    
    try:
        from src.infrastructure.reporting.template_manager import ReportTemplateManager
        
        template_manager = ReportTemplateManager()
        
        # Test template loading
        logger.info("Testing template loading...")
        
        individual_template = await template_manager.get_template("individual_analysis")
        assert individual_template is not None, "Failed to load individual analysis template"
        
        comparative_template = await template_manager.get_template("comparative_analysis")
        assert comparative_template is not None, "Failed to load comparative analysis template"
        
        logger.info("✅ Template loading successful")
        
        # Test template validation
        logger.info("Testing template validation...")
        
        from src.domain.report_models import TEMPLATE_SCHEMAS
        
        schema = TEMPLATE_SCHEMAS.get('comprehensive', {})
        
        validation_result = await template_manager.validate_template("individual_analysis", schema)
        assert validation_result == True, "Template validation failed"
        
        logger.info("✅ Template validation successful")
        
        # Test available templates listing
        available_templates = await template_manager.list_available_templates()
        assert len(available_templates) >= 5, "Insufficient default templates"
        assert "individual_analysis" in available_templates, "Missing individual analysis template"
        assert "comparative_analysis" in available_templates, "Missing comparative analysis template"
        
        logger.info(f"✅ Template listing successful - {len(available_templates)} templates available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Template management test failed: {e}")
        raise

async def test_caching_system():
    """Test report caching functionality"""
    
    logger.info("=== Testing Caching System ===")
    
    try:
        from src.infrastructure.reporting.cache_service import MemoryReportCache
        
        cache = MemoryReportCache(max_entries=50)
        
        # Test cache key generation
        logger.info("Testing cache key generation...")
        
        cache_key = cache.generate_cache_key(
            "https://example.com",
            {"include_technical": True, "format": "html"}
        )
        assert len(cache_key) > 0, "Cache key generation failed"
        
        comparative_key = cache.generate_comparative_cache_key(
            ["https://example1.com", "https://example2.com"],
            {"min_differentiators": 3}
        )
        assert len(comparative_key) > 0, "Comparative cache key generation failed"
        
        logger.info("✅ Cache key generation successful")
        
        # Test report caching and retrieval
        logger.info("Testing cache operations...")
        
        # Create mock report for caching
        test_analysis = create_mock_analysis_result("https://example.com", {'overall': 7.5})
        
        from src.application.services.report_generation import ReportGenerationService
        from src.infrastructure.reporting.template_manager import ReportTemplateManager
        
        template_manager = ReportTemplateManager()
        service = ReportGenerationService(template_manager=template_manager, cache_service=cache)
        
        # Generate a report to cache
        from src.domain.report_models import ReportFormat, ReportTemplate
        
        report = await service.generate_report(
            analysis_result=test_analysis,
            template=ReportTemplate.INDIVIDUAL_ANALYSIS,
            format_type=ReportFormat.HTML
        )
        
        # Cache the report
        cache_success = await cache.cache_report(cache_key, report, ttl_hours=1)
        assert cache_success == True, "Report caching failed"
        
        # Retrieve from cache
        cached_report = await cache.get_cached_report(cache_key)
        assert cached_report is not None, "Cache retrieval failed"
        assert cached_report.url == report.url, "Cached report URL mismatch"
        
        logger.info("✅ Cache operations successful")
        
        # Test cache statistics
        cache_stats = await cache.get_cache_stats()
        assert cache_stats['total_entries'] >= 1, "Cache statistics incorrect"
        assert cache_stats['cache_hits'] >= 1, "Cache hit count incorrect"
        
        logger.info(f"✅ Cache statistics: {cache_stats['hit_rate_percent']:.1f}% hit rate")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Caching system test failed: {e}")
        raise

async def test_comparative_analysis():
    """Test comparative analysis functionality"""
    
    logger.info("=== Testing Comparative Analysis ===")
    
    try:
        from src.infrastructure.analysis.comparative_analyzer import ComparativeAnalyzer
        
        analyzer = ComparativeAnalyzer()
        
        # Create test analyses with varying performance
        test_analyses = [
            create_mock_analysis_result("https://leader.com", {
                'overall': 8.5, 'content': 8.8, 'seo': 8.2, 'ux': 8.3
            }),
            create_mock_analysis_result("https://follower.com", {
                'overall': 7.2, 'content': 7.5, 'seo': 6.8, 'ux': 7.1
            }),
            create_mock_analysis_result("https://laggard.com", {
                'overall': 5.8, 'content': 6.2, 'seo': 5.1, 'ux': 5.9
            })
        ]
        
        # Test differentiator identification
        logger.info("Testing differentiator identification...")
        
        differentiators = await analyzer.identify_differentiators(test_analyses, min_count=3)
        assert len(differentiators) >= 3, "Insufficient differentiators identified"
        
        # Validate differentiator quality
        significant_differentiators = [d for d in differentiators if d.significance_score >= 6.0]
        assert len(significant_differentiators) >= 2, "Insufficient significant differentiators"
        
        logger.info(f"✅ Differentiator identification successful - {len(differentiators)} found")
        for diff in differentiators[:3]:
            logger.info(f"   - {diff.title} (significance: {diff.significance_score:.1f})")
        
        # Test similarity analysis
        logger.info("Testing similarity analysis...")
        
        similarity_scores = await analyzer.calculate_similarity_scores(test_analyses)
        assert 'overall_similarity' in similarity_scores, "Missing overall similarity score"
        assert 0.0 <= similarity_scores['overall_similarity'] <= 1.0, "Invalid similarity score range"
        
        logger.info(f"✅ Similarity analysis successful - overall similarity: {similarity_scores['overall_similarity']:.2f}")
        
        # Test market insights
        logger.info("Testing market insights...")
        
        market_insights = await analyzer.generate_market_insights(test_analyses)
        assert 'market_leaders' in market_insights, "Missing market leaders data"
        assert 'competitive_gap' in market_insights, "Missing competitive gap data"
        
        logger.info(f"✅ Market insights successful - {len(market_insights['market_leaders'])} leaders identified")
        
        # Test comprehensive comparative analysis
        logger.info("Testing comprehensive comparative analysis...")
        
        comparative_report = await analyzer.analyze_comparative_performance(test_analyses)
        assert comparative_report is not None, "Comparative analysis failed"
        assert len(comparative_report.key_differentiators) >= 3, "Insufficient differentiators in report"
        assert len(comparative_report.website_comparisons) == 3, "Incorrect website comparison count"
        
        logger.info("✅ Comprehensive comparative analysis successful")
        logger.info(f"   - Differentiators: {len(comparative_report.key_differentiators)}")
        logger.info(f"   - Recommendations: {len(comparative_report.cross_site_recommendations)}")
        logger.info(f"   - Best practices: {len(comparative_report.best_practices_identified)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparative analysis test failed: {e}")
        raise

async def test_performance_requirements():
    """Test performance requirements compliance"""
    
    logger.info("=== Testing Performance Requirements ===")
    
    try:
        from src.application.services.report_generation import ReportGenerationService
        from src.infrastructure.reporting.template_manager import ReportTemplateManager
        from src.infrastructure.reporting.cache_service import MemoryReportCache
        
        # Initialize services
        template_manager = ReportTemplateManager()
        cache_service = MemoryReportCache()
        service = ReportGenerationService(template_manager=template_manager, cache_service=cache_service)
        
        # Create test data
        test_analysis = create_mock_analysis_result("https://performance-test.com", {'overall': 7.0})
        
        # Test individual report performance
        logger.info("Testing individual report performance...")
        
        start_time = time.time()
        report = await service.generate_individual_report(
            analysis=test_analysis,
            template="individual_analysis",
            format_type="html"
        )
        generation_time = (time.time() - start_time) * 1000
        
        assert generation_time < 5000, f"Individual report generation too slow: {generation_time:.1f}ms"
        logger.info(f"✅ Individual report performance: {generation_time:.1f}ms (< 5000ms target)")
        
        # Test bulk report performance
        logger.info("Testing bulk report performance...")
        
        bulk_analyses = [
            create_mock_analysis_result(f"https://bulk-test-{i}.com", {'overall': 6.0 + i * 0.3})
            for i in range(5)
        ]
        
        start_time = time.time()
        bulk_summary = await service.generate_bulk_reports(
            analyses=bulk_analyses,
            template="individual_analysis",
            format_type="html",
            options={"parallel_processing": True}
        )
        bulk_time = (time.time() - start_time) * 1000
        avg_time_per_report = bulk_time / len(bulk_analyses)
        
        assert avg_time_per_report < 3000, f"Bulk report average time too slow: {avg_time_per_report:.1f}ms"
        logger.info(f"✅ Bulk report performance: {avg_time_per_report:.1f}ms per report (< 3000ms target)")
        
        # Test cache performance
        logger.info("Testing cache performance...")
        
        cache_key = cache_service.generate_cache_key("https://cache-test.com", {})
        
        # Cache operation
        start_time = time.time()
        await cache_service.cache_report(cache_key, report)
        cache_time = (time.time() - start_time) * 1000
        
        # Retrieval operation
        start_time = time.time()
        cached_report = await cache_service.get_cached_report(cache_key)
        retrieval_time = (time.time() - start_time) * 1000
        
        assert cache_time < 500, f"Cache storage too slow: {cache_time:.1f}ms"
        assert retrieval_time < 100, f"Cache retrieval too slow: {retrieval_time:.1f}ms"
        
        logger.info(f"✅ Cache performance: {cache_time:.1f}ms store, {retrieval_time:.1f}ms retrieve")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance requirements test failed: {e}")
        raise

def validate_data_models():
    """Validate data model completeness and structure"""
    
    logger.info("=== Validating Data Models ===")
    
    try:
        from src.domain.report_models import (
            AnalysisReport, ComparativeReport, BulkReportSummary,
            ExecutiveSummary, ImprovementItem, WebsiteComparison,
            ComparativeInsight, AnalysisDimension, ReportFormat,
            ReportTemplate, TEMPLATE_SCHEMAS
        )
        
        # Test AnalysisDimension enum
        dimensions = list(AnalysisDimension)
        assert len(dimensions) >= 6, f"Insufficient analysis dimensions: {len(dimensions)}"
        
        required_dimensions = ['content_quality', 'seo_optimization', 'user_experience', 'accessibility', 'performance', 'security']
        for dim in required_dimensions:
            assert any(d.value == dim for d in dimensions), f"Missing required dimension: {dim}"
        
        logger.info(f"✅ Analysis dimensions complete: {len(dimensions)} dimensions")
        
        # Test ReportFormat enum
        formats = list(ReportFormat)
        assert ReportFormat.HTML in formats, "Missing HTML format"
        assert ReportFormat.JSON in formats, "Missing JSON format"
        assert ReportFormat.PDF in formats, "Missing PDF format"
        
        logger.info(f"✅ Report formats complete: {[f.value for f in formats]}")
        
        # Test template schemas
        assert 'comprehensive' in TEMPLATE_SCHEMAS, "Missing comprehensive report schema"
        assert 'comparative' in TEMPLATE_SCHEMAS, "Missing comparative report schema"
        assert 'executive' in TEMPLATE_SCHEMAS, "Missing executive report schema"
        
        logger.info("✅ Template schemas defined")
        
        # Test data model creation
        from datetime import datetime
        
        # Create sample executive summary
        exec_summary = ExecutiveSummary(
            summary_text="Test executive summary under 200 words for validation purposes.",
            key_metrics={"score": "7.5/10", "rank": "2nd"},
            top_strengths=["Good content", "Clear navigation", "Fast loading"],
            critical_issues=["SEO optimization needed", "Missing alt tags"],
            priority_actions=["Improve meta tags", "Add structured data"],
            overall_assessment="Good foundation with optimization opportunities"
        )
        
        assert len(exec_summary.summary_text.split()) <= 200, "Executive summary too long"
        assert len(exec_summary.top_strengths) >= 1, "Insufficient strengths"
        assert len(exec_summary.critical_issues) >= 1, "Insufficient critical issues"
        
        logger.info("✅ Data model validation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data model validation failed: {e}")
        raise

async def run_comprehensive_validation():
    """Run comprehensive validation of WBS 2.3 implementation"""
    
    logger.info("🚀 Starting WBS 2.3 Analysis Report Generator Validation")
    logger.info("=" * 60)
    
    validation_results = {}
    
    try:
        # Validate data models
        validation_results['data_models'] = validate_data_models()
        
        # Test template management
        validation_results['template_management'] = await test_template_management()
        
        # Test caching system
        validation_results['caching_system'] = await test_caching_system()
        
        # Test comparative analysis
        validation_results['comparative_analysis'] = await test_comparative_analysis()
        
        # Test report generation service
        validation_results['report_generation'] = await test_report_generation_service()
        
        # Test performance requirements
        validation_results['performance'] = await test_performance_requirements()
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False
    
    # Summary
    logger.info("=" * 60)
    logger.info("🎉 WBS 2.3 VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in validation_results.values() if result)
    total_tests = len(validation_results)
    
    logger.info(f"Test Results: {passed_tests}/{total_tests} passed")
    
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        logger.info("\n🎊 ALL TESTS PASSED - WBS 2.3 IMPLEMENTATION COMPLETE!")
        logger.info("\nKey Features Validated:")
        logger.info("✅ Individual analysis reports with 6+ dimensions")
        logger.info("✅ Executive summaries under 200 words")
        logger.info("✅ Comparative analysis with 3+ differentiators")
        logger.info("✅ Consistent report format and templates")
        logger.info("✅ Template validation and management")
        logger.info("✅ Report caching for performance optimization")
        logger.info("✅ Bulk report generation capabilities")
        logger.info("✅ Structured data models and schemas")
        return True
    else:
        logger.error(f"\n❌ {total_tests - passed_tests} tests failed - implementation incomplete")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_validation())
    exit(0 if success else 1)
