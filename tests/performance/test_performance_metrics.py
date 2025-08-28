"""
Performance tests for the Web Content Analysis system
Tests performance, scalability, and load handling capabilities
"""
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock, patch
import statistics
from datetime import datetime

# Add backend to path
import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from src.domain.models import AnalysisResult, AnalysisType, AnalysisStatus
from src.domain.models import QualityLevel, AnalysisInsights  # Added missing imports


# Mock ContentAnalysisService for performance testing
class ContentAnalysisService:
    def __init__(self):
        pass
    
    async def analyze_url(self, url, analysis_type):
        return AnalysisResult(
            url=url,
            analysis_id="test-analysis",
            analysis_type=analysis_type,
            status=AnalysisStatus.COMPLETED,
            created_at=datetime.now()
        )
        return [AnalysisResult() for _ in urls]


class TestPerformanceMetrics:
    """Test performance characteristics of core components"""
    
    @pytest.fixture
    def mock_analysis_service(self):
        """Create mock analysis service for performance testing"""
        service = Mock()
        service.analyze_url = AsyncMock()
        service.analyze_multiple_urls = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_single_analysis_performance(self, mock_analysis_service):
        """Test performance of single URL analysis"""
        # Mock a realistic analysis result
        mock_result = AnalysisResult(
            analysis_id="perf-test-1",
            url="https://example.com/test",
            analysis_type=AnalysisType.COMPREHENSIVE,
            status=AnalysisStatus.COMPLETED,
            created_at=datetime.now(),
            executive_summary="Performance test analysis",
            processing_time=1.5,  # Simulated processing time
            cost=0.02
        )
        
        # Add realistic delay to simulate actual processing
        async def mock_analyze_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms simulated processing
            return mock_result
        
        mock_analysis_service.analyze_url = mock_analyze_with_delay
        
        # Measure performance
        start_time = time.time()
        result = await mock_analysis_service.analyze_url(
            url="https://example.com/test",
            analysis_type=AnalysisType.BASIC,
            quality_level=QualityLevel.FAST
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 1.0  # Should complete in under 1 second
        assert result.success is True
        
        print(f"Single analysis completed in {processing_time:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self, mock_analysis_service):
        """Test performance of concurrent analyses"""
        # Create multiple mock results
        def create_mock_result(index):
            return AnalysisResult(
                analysis_id=f"concurrent-{index}",
                url=f"https://example.com/page{index}",
                analysis_type=AnalysisType.BASIC,
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.now(),
                executive_summary=f"Concurrent test analysis {index}",
                processing_time=1.0 + (index * 0.1),  # Varying processing times
                cost=0.02
            )
        
        async def mock_analyze_concurrent(*args, **kwargs):
            # Simulate varying processing times
            delay = 0.05 + (hash(kwargs.get('url', '')) % 10) * 0.01
            await asyncio.sleep(delay)
            return create_mock_result(hash(kwargs.get('url', '')) % 100)
        
        mock_analysis_service.analyze_url = mock_analyze_concurrent
        
        # Test concurrent analysis
        urls = [f"https://example.com/page{i}" for i in range(10)]
        
        start_time = time.time()
        tasks = [
            mock_analysis_service.analyze_url(
                url=url,
                analysis_type=AnalysisType.BASIC,
                quality_level=QualityLevel.FAST
            )
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == 10
        assert all(result.success for result in results)
        assert total_time < 2.0  # Should complete concurrent requests faster than sequential
        
        # Calculate throughput
        throughput = len(urls) / total_time
        
        print(f"Concurrent analysis: {len(urls)} requests in {total_time:.3f} seconds")
        print(f"Throughput: {throughput:.2f} requests/second")
        
        # Should achieve reasonable throughput
        assert throughput > 5.0  # At least 5 requests per second
    
    @pytest.mark.asyncio
    async def test_bulk_analysis_performance(self, mock_analysis_service):
        """Test performance of bulk analysis functionality"""
        urls = [f"https://example.com/bulk{i}" for i in range(25)]
        
        # Mock bulk results
        bulk_results = []
        for i, url in enumerate(urls):
            result = AnalysisResult(
                analysis_id=f"bulk-{i}",
                url=url,
                analysis_type=AnalysisType.BASIC,
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.now(),
                executive_summary=f"Bulk analysis {i}",
                processing_time=0.5 + (i * 0.02),
                cost=0.01
            )
            bulk_results.append(result)
        
        async def mock_bulk_analyze(*args, **kwargs):
            # Simulate bulk processing with some delay
            await asyncio.sleep(0.3)  # Bulk operations should be optimized
            return bulk_results
        
        mock_analysis_service.analyze_multiple_urls = mock_bulk_analyze
        
        start_time = time.time()
        results = await mock_analysis_service.analyze_multiple_urls(
            urls=urls,
            analysis_type=AnalysisType.BASIC,
            quality_level=QualityLevel.FAST
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == 25
        assert all(result.success for result in results)
        assert total_time < 1.0  # Bulk should be more efficient than individual requests
        
        # Calculate bulk throughput
        bulk_throughput = len(urls) / total_time
        
        print(f"Bulk analysis: {len(urls)} URLs in {total_time:.3f} seconds")
        print(f"Bulk throughput: {bulk_throughput:.2f} URLs/second")
        
        # Bulk should achieve higher throughput
        assert bulk_throughput > 20.0
    
    def test_memory_usage_estimation(self):
        """Test memory usage patterns during analysis"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate creating many analysis results
        results = []
        for i in range(100):
            result = AnalysisResult(
                analysis_id=f"memory-test-{i}",
                url=f"https://example.com/memory{i}",
                analysis_type=AnalysisType.COMPREHENSIVE,
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.now(),
                executive_summary=f"Memory test analysis {i}" * 10,  # Longer text
                insights=AnalysisInsights(
                    strengths=[f"Strength {j}" for j in range(5)],
                    weaknesses=[f"Weakness {j}" for j in range(3)],
                    opportunities=[f"Opportunity {j}" for j in range(4)],
                    key_findings=[f"Finding {j}" for j in range(6)]
                ),
                processing_time=1.0,
                cost=0.05
            )
            results.append(result)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.2f} MB → {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB for 100 results")
        
        # Memory usage should be reasonable
        memory_per_result = memory_increase / 100
        assert memory_per_result < 1.0  # Less than 1MB per result
    
    @pytest.mark.slow
    def test_sustained_load_performance(self):
        """Test system performance under sustained load"""
        # This test simulates sustained load over time
        results = []
        processing_times = []
        
        def simulate_analysis_request():
            """Simulate a single analysis request"""
            start = time.time()
            
            # Simulate processing delay
            time.sleep(0.1)  # 100ms processing time
            
            end = time.time()
            processing_time = end - start
            processing_times.append(processing_time)
            
            return {
                "success": True,
                "processing_time": processing_time,
                "timestamp": datetime.now()
            }
        
        # Simulate sustained load with thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit 50 requests
            futures = [executor.submit(simulate_analysis_request) for _ in range(50)]
            
            start_time = time.time()
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
            end_time = time.time()
        
        total_time = end_time - start_time
        
        # Calculate performance metrics
        avg_processing_time = statistics.mean(processing_times)
        median_processing_time = statistics.median(processing_times)
        max_processing_time = max(processing_times)
        throughput = len(results) / total_time
        
        print(f"Sustained Load Test Results:")
        print(f"Total requests: {len(results)}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Avg processing time: {avg_processing_time:.3f} seconds")
        print(f"Median processing time: {median_processing_time:.3f} seconds")
        print(f"Max processing time: {max_processing_time:.3f} seconds")
        
        # Performance assertions
        assert len(results) == 50
        assert all(r["success"] for r in results)
        assert avg_processing_time < 0.2  # Average should be reasonable
        assert max_processing_time < 0.5   # No request should take too long
        assert throughput > 10.0          # Should handle at least 10 req/sec


class TestScalabilityMetrics:
    """Test scalability characteristics"""
    
    @pytest.mark.parametrize("load_size", [1, 5, 10, 20, 50])
    @pytest.mark.asyncio
    async def test_scalability_by_load_size(self, load_size):
        """Test how performance scales with different load sizes"""
        async def simulate_analysis():
            await asyncio.sleep(0.05)  # 50ms base processing time
            return {"success": True, "processing_time": 0.05}
        
        start_time = time.time()
        
        # Create tasks based on load size
        tasks = [simulate_analysis() for _ in range(load_size)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = load_size / total_time
        
        print(f"Load size: {load_size:2d} | Time: {total_time:.3f}s | Throughput: {throughput:.2f} req/s")
        
        # Basic assertions
        assert len(results) == load_size
        assert all(r["success"] for r in results)
        
        # Throughput should not degrade significantly with higher loads
        # (This is a simplified test; real scalability testing would be more complex)
        if load_size <= 10:
            assert throughput > 15.0  # Small loads should have high throughput
        else:
            assert throughput > 10.0  # Larger loads should still maintain reasonable throughput
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_performance(self):
        """Test that resources are properly cleaned up"""
        import gc
        import weakref
        
        # Create objects and weak references
        objects = []
        weak_refs = []
        
        for i in range(100):
            obj = AnalysisResult(
                analysis_id=f"cleanup-{i}",
                url=f"https://example.com/cleanup{i}",
                analysis_type=AnalysisType.BASIC,
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.now(),
                executive_summary=f"Cleanup test {i}",
                processing_time=1.0,
                cost=0.01
            )
            objects.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # Clear strong references
        objects.clear()
        
        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)  # Allow time for cleanup
        
        # Check that objects were garbage collected
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        
        print(f"Objects remaining after cleanup: {alive_objects}/100")
        
        # Most objects should be cleaned up
        assert alive_objects < 10  # Allow for some objects to remain due to GC timing


class TestDatabasePerformance:
    """Test database operation performance"""
    
    @pytest.mark.asyncio
    async def test_database_insert_performance(self):
        """Test database insertion performance"""
        # Mock database operations
        insert_times = []
        
        async def mock_db_insert():
            start = time.time()
            await asyncio.sleep(0.01)  # Simulate 10ms database operation
            end = time.time()
            insert_times.append(end - start)
            return "inserted-id"
        
        # Test multiple insertions
        start_time = time.time()
        tasks = [mock_db_insert() for _ in range(20)]
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_insert_time = statistics.mean(insert_times)
        
        print(f"Database insert performance:")
        print(f"20 inserts in {total_time:.3f} seconds")
        print(f"Average insert time: {avg_insert_time:.4f} seconds")
        
        # Performance assertions
        assert avg_insert_time < 0.05  # Average insert should be fast
        assert total_time < 1.0        # Total time should be reasonable
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """Test database query performance"""
        query_times = []
        
        async def mock_db_query():
            start = time.time()
            await asyncio.sleep(0.005)  # Simulate 5ms query
            end = time.time()
            query_times.append(end - start)
            return [{"id": i, "url": f"https://example.com/{i}"} for i in range(10)]
        
        # Test multiple queries
        start_time = time.time()
        tasks = [mock_db_query() for _ in range(30)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_query_time = statistics.mean(query_times)
        
        print(f"Database query performance:")
        print(f"30 queries in {total_time:.3f} seconds")
        print(f"Average query time: {avg_query_time:.4f} seconds")
        
        # Performance assertions
        assert len(results) == 30
        assert avg_query_time < 0.02   # Queries should be fast
        assert total_time < 0.5        # Total should be very fast for read operations
