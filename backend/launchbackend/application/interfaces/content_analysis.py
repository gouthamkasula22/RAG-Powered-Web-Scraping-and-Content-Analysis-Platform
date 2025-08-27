"""
Interface for content analysis service
"""
from abc import ABC, abstractmethod
from typing import Optional
from backend.launchbackend.domain.models import AnalysisResult, AnalysisType


class IContentAnalysisService(ABC):
    """Interface for content analysis service"""
    
    @abstractmethod
    async def analyze_url(self, url: str, analysis_type: AnalysisType) -> AnalysisResult:
        """Analyze content from URL"""
        pass
    
    @abstractmethod
    async def analyze_content(self, content: str, url: str, analysis_type: AnalysisType) -> AnalysisResult:
        """Analyze provided content"""
        pass
    
    @abstractmethod
    async def get_analysis_status(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis status by ID"""
        pass
    
    @abstractmethod
    def estimate_analysis_cost(self, content_length: int, analysis_type: AnalysisType) -> float:
        """Estimate cost for analysis"""
        pass
