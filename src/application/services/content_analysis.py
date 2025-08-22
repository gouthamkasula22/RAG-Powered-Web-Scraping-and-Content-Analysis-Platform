"""
Production content analysis service implementation
"""
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import re
import json

from ...domain.models import (
    AnalysisResult, AnalysisStatus, AnalysisType, AnalysisMetrics, 
    AnalysisInsights, ScrapedContent, URLInfo, ContentMetrics,
    ContentType, ScrapingStatus
)
from ...application.interfaces.content_analysis import IContentAnalysisService
from ...application.interfaces.llm import AnalysisRequest
from ...application.interfaces.scraping import IWebScraper

logger = logging.getLogger(__name__)


class ContentAnalysisService(IContentAnalysisService):
    """Production content analysis service"""
    
    def __init__(self, 
                 scraping_service: IWebScraper,
                 llm_service):  # Use Any to avoid circular import
        """Initialize content analysis service
        
        Args:
            scraping_service: Web scraping service
            llm_service: LLM service for content analysis
        """
        self.scraping_service = scraping_service
        self.llm_service = llm_service
        self.analysis_cache: Dict[str, AnalysisResult] = {}
    
    async def analyze_url(self, url: str, analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE) -> AnalysisResult:
        """Complete URL analysis pipeline"""
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize analysis result
        result = AnalysisResult(
            url=url,
            analysis_id=analysis_id,
            analysis_type=analysis_type,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Cache the initial result
        self.analysis_cache[analysis_id] = result
        
        try:
            logger.info(f"🚀 Starting analysis for URL: {url}")
            
            # Step 1: Web Scraping
            result.status = AnalysisStatus.SCRAPING
            
            # Create scraping request
            from ...domain.models import ScrapingRequest
            scraping_request = ScrapingRequest(url=url)
            scraping_result = await self.scraping_service.scrape_content(scraping_request)
            
            if not scraping_result.is_success:
                result.status = AnalysisStatus.FAILED
                result.error_message = f"Scraping failed: {scraping_result.error_message}"
                result.completed_at = datetime.now()
                result.processing_time = time.time() - start_time
                return result
            
            result.scraped_content = scraping_result.content
            
            # Step 2: Content Analysis
            result.status = AnalysisStatus.ANALYZING
            analysis_result = await self._perform_llm_analysis(
                scraping_result.content, 
                url, 
                analysis_type
            )
            
            # Step 3: Process and Structure Results
            await self._process_analysis_results(result, analysis_result)
            
            # Step 4: Finalize
            result.status = AnalysisStatus.COMPLETED
            result.completed_at = datetime.now()
            result.processing_time = time.time() - start_time
            
            logger.info(f"✅ Analysis completed for {url} in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {url}: {e}")
            result.status = AnalysisStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.processing_time = time.time() - start_time
            return result
    
    async def analyze_content(self, content: str, url: str, analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE) -> AnalysisResult:
        """Analyze provided content directly"""
        
        # Create mock scraped content  
        url_info = URLInfo.from_url(url) if url else URLInfo(url="direct-content", domain="direct")
        
        # Use ContentMetrics.calculate method
        content_metrics = ContentMetrics.calculate(
            content=content,
            links=[],  # No links in direct content
            headings=re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)  # Extract markdown headings
        )
        
        scraped_content = ScrapedContent(
            url_info=url_info,
            title="Direct Content Analysis",
            headings=re.findall(r'^#+\s+(.+)$', content, re.MULTILINE),  # Extract markdown headings
            main_content=content,
            links=[],  # No links in direct content
            meta_description=None,
            meta_keywords=[],
            content_type=ContentType.ARTICLE,
            metrics=content_metrics,
            scraped_at=datetime.now(),
            status=ScrapingStatus.SUCCESS
        )
        
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        result = AnalysisResult(
            url=url or "direct-content",
            analysis_id=analysis_id,
            analysis_type=analysis_type,
            status=AnalysisStatus.ANALYZING,
            created_at=datetime.now(),
            scraped_content=scraped_content
        )
        
        try:
            # Perform LLM analysis
            analysis_result = await self._perform_llm_analysis(scraped_content, url or "direct-content", analysis_type)
            
            # Process results
            await self._process_analysis_results(result, analysis_result)
            
            result.status = AnalysisStatus.COMPLETED
            result.completed_at = datetime.now()
            result.processing_time = time.time() - start_time
            
            logger.info(f"✅ Content analysis completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Content analysis failed: {e}")
            result.status = AnalysisStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.processing_time = time.time() - start_time
            return result
    
    async def _perform_llm_analysis(self, scraped_content: ScrapedContent, url: str, analysis_type: AnalysisType):
        """Perform LLM analysis on scraped content"""
        
        # Prepare content for analysis
        analysis_content = self._prepare_content_for_analysis(scraped_content)
        
        # Create LLM request with updated parameters for Gemini-2.0-Flash
        request = AnalysisRequest(
            content=analysis_content,
            analysis_type=analysis_type.value,
            max_cost=0.05,  # Default cost limit
            quality_preference="balanced"
        )
        
        # Perform analysis
        llm_response = await self.llm_service.analyze_content(request)
        
        if not llm_response.success:
            raise Exception(f"LLM analysis failed: {llm_response.error_message}")
        
        return llm_response
    
    def _prepare_content_for_analysis(self, scraped_content: ScrapedContent) -> str:
        """Prepare scraped content for LLM analysis with improved formatting"""
        
        # Create comprehensive analysis context
        analysis_text = f"""
WEBSITE CONTENT ANALYSIS REQUEST

=== METADATA ===
URL: {scraped_content.url_info.url}
Domain: {scraped_content.url_info.domain}
Title: {scraped_content.title}
Extracted: {scraped_content.scraped_at}

=== CONTENT METRICS ===
• Content Length: {len(scraped_content.main_content):,} characters
• Word Count: {scraped_content.metrics.word_count:,} words
• Paragraphs: {scraped_content.metrics.paragraph_count}
• Sentences: {scraped_content.metrics.sentence_count}
• Avg Words/Sentence: {scraped_content.metrics.word_count / max(scraped_content.metrics.sentence_count, 1):.1f}
• Links: {scraped_content.metrics.link_count}
• Headings: {scraped_content.metrics.heading_count}
• Reading Time: {scraped_content.metrics.reading_time_minutes:.1f} minutes

=== MAIN CONTENT ===
{scraped_content.main_content}

=== TECHNICAL METADATA ===
Meta Description: {scraped_content.meta_description or "None"}
Meta Keywords: {', '.join(scraped_content.meta_keywords) if scraped_content.meta_keywords else "None"}
Content Type: {scraped_content.content_type.value}
Status: {scraped_content.status.value}

=== ANALYSIS INSTRUCTIONS ===
Please provide a comprehensive analysis focusing on:
1. Content quality and structure
2. SEO optimization potential
3. User experience factors
4. Readability and engagement
5. Specific actionable recommendations
"""
        return analysis_text
    
    async def _process_analysis_results(self, result: AnalysisResult, llm_response):
        """Process and structure LLM analysis results"""
        result.llm_response = llm_response
        result.cost = getattr(llm_response, 'cost', 0.0)
        result.provider_used = getattr(llm_response, 'provider', 'unknown').value if hasattr(getattr(llm_response, 'provider', 'unknown'), 'value') else str(getattr(llm_response, 'provider', 'unknown'))
        
        # Extract structured data from LLM response
        try:
            structured_data = self._extract_structured_data(llm_response.content)
            
            # Set executive summary
            result.executive_summary = structured_data.get('executive_summary', 
                self._extract_executive_summary(llm_response.content))
            
            # Extract metrics
            result.metrics = self._extract_metrics(structured_data, llm_response.content)
            
            # Extract insights
            result.insights = self._extract_insights(structured_data, llm_response.content)
            
            # Store detailed analysis
            result.detailed_analysis = structured_data
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract structured data: {e}")
            result.warnings.append(f"Structured data extraction failed: {e}")
            result.executive_summary = self._extract_executive_summary(llm_response.content)
    
    def _extract_structured_data(self, llm_content: str) -> Dict[str, Any]:
        """Extract structured data from LLM response"""
        structured_data = {}
        
        # Extract sections based on markdown headers
        sections = re.split(r'\n## ', llm_content)
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n', 1)
            if len(lines) < 2:
                continue
                
            header = lines[0].replace('#', '').strip()
            content = lines[1].strip()
            
            # Clean header for key
            key = header.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
            structured_data[key] = content
        
        return structured_data
    
    def _extract_executive_summary(self, content: str) -> str:
        """Extract executive summary from analysis"""
        # Look for executive summary section
        summary_patterns = [
            r'## 🎯 Executive Summary\n(.*?)(?=\n##|$)',
            r'## Executive Summary\n(.*?)(?=\n##|$)',
            r'### Executive Summary\n(.*?)(?=\n##|$)',
            r'## Summary\n(.*?)(?=\n##|$)',
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: take first substantial paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        return paragraphs[0] if paragraphs else "Analysis completed successfully."
    
    def _extract_metrics(self, structured_data: Dict, content: str) -> AnalysisMetrics:
        """Extract quantified metrics from analysis"""
        # Look for scores in the content
        score_pattern = r'(\d+(?:\.\d+)?)/10'
        scores = re.findall(score_pattern, content)
        
        # Default scores
        default_score = 7.0
        
        try:
            # Try to extract specific scores
            content_quality = float(scores[0]) if len(scores) > 0 else default_score
            seo_score = float(scores[1]) if len(scores) > 1 else default_score
            ux_score = float(scores[2]) if len(scores) > 2 else default_score
            readability = float(scores[3]) if len(scores) > 3 else default_score
            engagement = float(scores[4]) if len(scores) > 4 else default_score
            
            overall_score = (content_quality + seo_score + ux_score + readability + engagement) / 5
            
            return AnalysisMetrics(
                content_quality_score=content_quality,
                seo_score=seo_score,
                ux_score=ux_score,
                readability_score=readability,
                engagement_score=engagement,
                overall_score=overall_score
            )
            
        except (ValueError, IndexError):
            # Fallback to default scores
            return AnalysisMetrics(
                content_quality_score=default_score,
                seo_score=default_score,
                ux_score=default_score,
                readability_score=default_score,
                engagement_score=default_score,
                overall_score=default_score
            )
    
    def _extract_insights(self, structured_data: Dict, content: str) -> AnalysisInsights:
        """Extract insights from analysis"""
        # Extract different types of insights
        strengths = self._extract_list_items(content, ['strengths', 'strong points', 'positives'])
        weaknesses = self._extract_list_items(content, ['weaknesses', 'areas for improvement', 'issues'])
        opportunities = self._extract_list_items(content, ['opportunities', 'potential improvements'])
        threats = self._extract_list_items(content, ['threats', 'risks', 'concerns'])
        findings = self._extract_list_items(content, ['key findings', 'findings', 'observations'])
        recommendations = self._extract_list_items(content, ['recommendations', 'suggestions', 'action items'])
        
        return AnalysisInsights(
            strengths=strengths[:5],  # Limit to top 5
            weaknesses=weaknesses[:5],
            opportunities=opportunities[:5],
            threats=threats[:3],
            key_findings=findings[:5],
            recommendations=recommendations[:8]
        )
    
    def _extract_list_items(self, content: str, keywords: list) -> list:
        """Extract list items from content based on keywords"""
        items = []
        
        for keyword in keywords:
            # Look for sections with these keywords
            pattern = rf'(?i).*{keyword}.*?\n((?:[-*•]\s.*\n?)*)'
            matches = re.findall(pattern, content, re.MULTILINE)
            
            for match in matches:
                # Extract bullet points
                bullets = re.findall(r'[-*•]\s(.+)', match)
                items.extend([bullet.strip() for bullet in bullets])
        
        # Remove duplicates and empty items
        unique_items = []
        for item in items:
            if item and item not in unique_items:
                unique_items.append(item)
        
        return unique_items
    
    async def get_analysis_status(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis status by ID"""
        return self.analysis_cache.get(analysis_id)
    
    def estimate_analysis_cost(self, content_length: int, analysis_type: AnalysisType) -> float:
        """Estimate cost for analysis - Updated for Gemini-2.0-Flash"""
        # Rough estimation based on content length
        tokens = content_length // 4  # ~4 chars per token
        
        # With Gemini-2.0-Flash, most content will be free
        if tokens <= 800000:  # Within Gemini's practical limits
            return 0.0  # Gemini free tier
        else:
            # Claude pricing for very large content
            return (tokens * 0.00025 / 1000) + 0.01  # Input + output cost
