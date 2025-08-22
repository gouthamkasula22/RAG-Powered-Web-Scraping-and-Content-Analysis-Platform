"""
Report Cache Service implementing WBS 2.3 caching requirements.
Handles report caching for performance optimization and bulk processing.
"""
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pickle

from ...application.interfaces.report_generation import IReportCache, ReportPriority
from ...domain.report_models import AnalysisReport, ComparativeReport, BulkReportSummary, ReportTemplate, ReportFormat

logger = logging.getLogger(__name__)


class ReportCacheError(Exception):
    """Exception raised when cache operations fail"""
    pass


class ReportCache(IReportCache):
    """
    Production report cache service with TTL and priority management.
    Optimizes report generation performance through intelligent caching.
    """
    
    def __init__(self, cache_directory: str = "cache/reports", max_cache_size_mb: int = 500):
        """Initialize report cache with configuration"""
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.default_ttl_hours = 24
        
        # Cache metadata tracking
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        logger.info(f"Report cache initialized: {self.cache_dir}")
    
    async def get_cached_report(self, cache_key: str) -> Optional[Union[AnalysisReport, ComparativeReport]]:
        """Retrieve cached report if available and valid"""
        
        self._stats['total_requests'] += 1
        
        try:
            # Check if cache entry exists
            if cache_key not in self._metadata:
                self._stats['misses'] += 1
                return None
            
            entry_meta = self._metadata[cache_key]
            
            # Check TTL
            if self._is_expired(entry_meta):
                await self._remove_cache_entry(cache_key)
                self._stats['misses'] += 1
                return None
            
            # Load cached report
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                # Metadata exists but file is missing - clean up
                await self._remove_cache_entry(cache_key)
                self._stats['misses'] += 1
                return None
            
            with open(cache_file, 'rb') as f:
                report = pickle.load(f)
            
            # Update access time
            entry_meta['last_accessed'] = datetime.now().isoformat()
            entry_meta['access_count'] += 1
            
            self._save_metadata()
            self._stats['hits'] += 1
            
            logger.debug(f"Cache hit for key: {cache_key}")
            return report
            
        except Exception as e:
            logger.error(f"Cache retrieval failed for key {cache_key}: {e}")
            # Clean up corrupted entry
            await self._remove_cache_entry(cache_key)
            self._stats['misses'] += 1
            return None
    
    async def cache_report(self, cache_key: str, report: Union[AnalysisReport, ComparativeReport], 
                          ttl_hours: Optional[int] = None, priority: ReportPriority = ReportPriority.NORMAL) -> bool:
        """Cache report with TTL and priority"""
        
        try:
            # Calculate expiry time
            ttl = ttl_hours or self.default_ttl_hours
            expires_at = datetime.now() + timedelta(hours=ttl)
            
            # Serialize report
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(report, f)
            
            # Get file size
            file_size = cache_file.stat().st_size
            
            # Update metadata
            self._metadata[cache_key] = {
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at.isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0,
                'file_size_bytes': file_size,
                'priority': priority.value,
                'report_type': type(report).__name__
            }
            
            self._save_metadata()
            
            # Check cache size and evict if necessary
            await self._enforce_cache_limits()
            
            logger.debug(f"Cached report with key: {cache_key}, size: {file_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Cache storage failed for key {cache_key}: {e}")
            # Clean up partial files
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            return False
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern"""
        
        removed_count = 0
        keys_to_remove = []
        
        if pattern:
            # Remove entries matching pattern
            for key in self._metadata.keys():
                if pattern in key:
                    keys_to_remove.append(key)
        else:
            # Remove all entries
            keys_to_remove = list(self._metadata.keys())
        
        for key in keys_to_remove:
            success = await self._remove_cache_entry(key)
            if success:
                removed_count += 1
        
        logger.info(f"Invalidated {removed_count} cache entries")
        return removed_count
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_size = sum(entry['file_size_bytes'] for entry in self._metadata.values())
        hit_rate = (self._stats['hits'] / max(self._stats['total_requests'], 1)) * 100
        
        return {
            'total_entries': len(self._metadata),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'max_size_mb': round(self.max_cache_size_bytes / (1024 * 1024), 2),
            'hit_rate_percent': round(hit_rate, 2),
            'cache_hits': self._stats['hits'],
            'cache_misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'total_requests': self._stats['total_requests'],
            'oldest_entry': self._get_oldest_entry_age(),
            'entry_breakdown': self._get_entry_breakdown()
        }
    
    def generate_cache_key(self, url: str, analysis_options: Dict[str, Any]) -> str:
        """Generate deterministic cache key from URL and options"""
        
        # Create consistent string representation
        options_str = json.dumps(analysis_options, sort_keys=True)
        combined = f"{url}|{options_str}"
        
        # Generate hash
        cache_key = hashlib.sha256(combined.encode()).hexdigest()[:32]
        
        return f"report_{cache_key}"
    
    def generate_comparative_cache_key(self, urls: List[str], options: Dict[str, Any]) -> str:
        """Generate cache key for comparative analysis"""
        
        # Sort URLs for consistency
        sorted_urls = sorted(urls)
        urls_str = "|".join(sorted_urls)
        options_str = json.dumps(options, sort_keys=True)
        combined = f"comparative|{urls_str}|{options_str}"
        
        cache_key = hashlib.sha256(combined.encode()).hexdigest()[:32]
        
        return f"comp_{cache_key}"
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries"""
        
        expired_keys = []
        
        for key, entry_meta in self._metadata.items():
            if self._is_expired(entry_meta):
                expired_keys.append(key)
        
        removed_count = 0
        for key in expired_keys:
            success = await self._remove_cache_entry(key)
            if success:
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    async def _enforce_cache_limits(self):
        """Enforce cache size limits through LRU eviction"""
        
        total_size = sum(entry['file_size_bytes'] for entry in self._metadata.values())
        
        if total_size <= self.max_cache_size_bytes:
            return
        
        # Sort entries by last accessed time (LRU first)
        entries_by_access = sorted(
            self._metadata.items(),
            key=lambda x: (
                x[1].get('priority', 'normal') == 'low',  # Low priority first
                datetime.fromisoformat(x[1]['last_accessed'])  # Then by access time
            )
        )
        
        evicted_count = 0
        
        for key, entry_meta in entries_by_access:
            if total_size <= self.max_cache_size_bytes:
                break
            
            # Don't evict high priority entries unless absolutely necessary
            if entry_meta.get('priority') == 'high' and total_size < self.max_cache_size_bytes * 1.2:
                continue
            
            file_size = entry_meta['file_size_bytes']
            success = await self._remove_cache_entry(key)
            
            if success:
                total_size -= file_size
                evicted_count += 1
                self._stats['evictions'] += 1
        
        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count} cache entries to enforce size limits")
    
    async def _remove_cache_entry(self, cache_key: str) -> bool:
        """Remove cache entry and associated files"""
        
        try:
            # Remove file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove metadata
            if cache_key in self._metadata:
                del self._metadata[cache_key]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")
            return False
    
    def _is_expired(self, entry_meta: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        
        try:
            expires_at = datetime.fromisoformat(entry_meta['expires_at'])
            return datetime.now() > expires_at
        except (KeyError, ValueError):
            # Invalid metadata - consider expired
            return True
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                logger.debug(f"Loaded cache metadata: {len(self._metadata)} entries")
            else:
                self._metadata = {}
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            self._metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_oldest_entry_age(self) -> str:
        """Get age of oldest cache entry"""
        
        if not self._metadata:
            return "No entries"
        
        oldest_time = min(
            datetime.fromisoformat(entry['created_at']) 
            for entry in self._metadata.values()
        )
        
        age = datetime.now() - oldest_time
        return f"{age.days}d {age.seconds // 3600}h"
    
    def _get_entry_breakdown(self) -> Dict[str, int]:
        """Get breakdown of cache entries by type"""
        
        breakdown = {}
        
        for entry in self._metadata.values():
            report_type = entry.get('report_type', 'Unknown')
            breakdown[report_type] = breakdown.get(report_type, 0) + 1
        
        return breakdown


class MemoryReportCache(IReportCache):
    """
    In-memory report cache for testing and development.
    Provides fast access but limited persistence.
    """
    
    def __init__(self, max_entries: int = 100):
        """Initialize memory cache with entry limit"""
        self.max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    async def get_cached_report(self, cache_key: str) -> Optional[Union[AnalysisReport, ComparativeReport]]:
        """Get report from memory cache"""
        
        self._stats['total_requests'] += 1
        
        if cache_key not in self._cache:
            self._stats['misses'] += 1
            return None
        
        entry = self._cache[cache_key]
        
        # Check TTL
        if datetime.now() > entry['expires_at']:
            del self._cache[cache_key]
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._stats['misses'] += 1
            return None
        
        # Update access order
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
        
        self._stats['hits'] += 1
        return entry['report']
    
    async def cache_report(self, cache_key: str, report: Union[AnalysisReport, ComparativeReport], 
                          ttl_hours: Optional[int] = None, priority: ReportPriority = ReportPriority.NORMAL) -> bool:
        """Cache report in memory"""
        
        try:
            # Calculate expiry
            ttl = ttl_hours or 24
            expires_at = datetime.now() + timedelta(hours=ttl)
            
            # Store entry
            self._cache[cache_key] = {
                'report': report,
                'expires_at': expires_at,
                'priority': priority,
                'created_at': datetime.now()
            }
            
            # Update access order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            
            # Enforce size limits
            await self._enforce_limits()
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cache storage failed: {e}")
            return False
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate memory cache entries"""
        
        if pattern:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
        else:
            keys_to_remove = list(self._cache.keys())
        
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        return len(keys_to_remove)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        
        hit_rate = (self._stats['hits'] / max(self._stats['total_requests'], 1)) * 100
        
        return {
            'total_entries': len(self._cache),
            'max_entries': self.max_entries,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_hits': self._stats['hits'],
            'cache_misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'total_requests': self._stats['total_requests']
        }
    
    def generate_cache_key(self, url: str, analysis_options: Dict[str, Any]) -> str:
        """Generate cache key for memory cache"""
        
        options_str = json.dumps(analysis_options, sort_keys=True)
        combined = f"{url}|{options_str}"
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def generate_comparative_cache_key(self, urls: List[str], options: Dict[str, Any]) -> str:
        """Generate comparative cache key"""
        
        sorted_urls = sorted(urls)
        urls_str = "|".join(sorted_urls)
        options_str = json.dumps(options, sort_keys=True)
        combined = f"comp|{urls_str}|{options_str}"
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired entries from memory"""
        
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items() 
            if now > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        return len(expired_keys)
    
    async def _enforce_limits(self):
        """Enforce memory cache size limits"""
        
        while len(self._cache) > self.max_entries:
            # Remove least recently used
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
                    self._stats['evictions'] += 1
            else:
                break
    
    # Interface implementation methods
    async def store_report(
        self,
        analysis_id: str,
        template: ReportTemplate,
        format_type: ReportFormat,
        report_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store report using interface signature"""
        cache_key = f"{analysis_id}_{template.value}_{format_type.value}"
        mock_report = AnalysisReport(
            analysis_id=analysis_id,
            url=analysis_id,  # Use analysis_id as URL for simplicity
            content=report_data,
            generated_at=datetime.now(),
            format_type=format_type
        )
        return await self.cache_report(cache_key, mock_report, ttl_seconds)
    
    async def retrieve_report(
        self,
        analysis_id: str,
        template: ReportTemplate,
        format_type: ReportFormat
    ) -> Optional[Dict[str, Any]]:
        """Retrieve report using interface signature"""
        cache_key = f"{analysis_id}_{template.value}_{format_type.value}"
        report = await self.get_cached_report(cache_key)
        return report.content if report else None
