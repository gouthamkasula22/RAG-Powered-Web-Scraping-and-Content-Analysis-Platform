"""
Report Template Manager implementing WBS 2.3 template management requirements.
Handles Jinja2 template loading, validation, and custom template support.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from ...application.interfaces.report_generation import (
    IReportTemplateManager, ReportTemplate, ReportFormat
)
from ...domain.report_models import TEMPLATE_SCHEMAS

logger = logging.getLogger(__name__)


class TemplateValidationError(Exception):
    """Exception raised when template validation fails"""
    pass


class ReportTemplateManager(IReportTemplateManager):
    """
    Production template manager for report generation.
    Manages Jinja2 templates with validation and caching.
    """
    
    def __init__(self, template_directory: str = "templates"):
        """Initialize template manager with template directory"""
        self.template_dir = Path(template_directory)
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True
        )
        
        # Template cache
        self._template_cache: Dict[str, Template] = {}
        
        # Initialize default templates
        self._create_default_templates()
    
    async def get_template(self, template_name: str) -> Template:
        """Get template by name with caching"""
        
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        try:
            # Load template from filesystem
            template = self.env.get_template(f"{template_name}.html")
            
            # Cache template
            self._template_cache[template_name] = template
            
            logger.debug(f"Loaded template: {template_name}")
            return template
            
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            # Return default template
            return await self._get_default_template(template_name)
    
    async def validate_template(self, template_name: str, schema: Dict[str, Any]) -> bool:
        """Validate template against expected schema"""
        
        try:
            template = await self.get_template(template_name)
            
            # Test render with sample data based on schema
            test_data = self._generate_test_data(schema)
            rendered = template.render(**test_data)
            
            # Basic validation - check if template rendered without errors
            if not rendered.strip():
                raise TemplateValidationError(f"Template {template_name} produces empty output")
            
            # Format-specific validation
            if template_name.endswith('_json'):
                json.loads(rendered)  # Validate JSON structure
            
            logger.info(f"Template validation successful: {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed for {template_name}: {e}")
            raise TemplateValidationError(f"Template {template_name} validation failed: {e}")
    
    async def list_available_templates(self) -> List[str]:
        """List all available templates"""
        
        templates = []
        
        # Scan template directory
        for template_file in self.template_dir.glob("*.html"):
            template_name = template_file.stem
            templates.append(template_name)
        
        # Add built-in templates
        built_in_templates = [
            "individual_analysis",
            "comparative_analysis", 
            "executive_summary",
            "technical_report",
            "bulk_summary"
        ]
        
        for template in built_in_templates:
            if template not in templates:
                templates.append(template)
        
        return sorted(templates)
    
    def get_available_templates(self) -> List[str]:
        """Synchronous version for interface compatibility"""
        import asyncio
        return asyncio.run(self.list_available_templates())
    
    def load_template(self, template_type: ReportTemplate) -> str:
        """Load template content by template type"""
        from src.domain.report_models import ReportTemplate
        
        template_mapping = {
            ReportTemplate.COMPREHENSIVE: "individual_analysis",
            ReportTemplate.EXECUTIVE: "executive_summary", 
            ReportTemplate.COMPARATIVE: "comparative_analysis",
            ReportTemplate.TECHNICAL: "technical_report",
            ReportTemplate.BULK: "bulk_summary"
        }
        
        template_name = template_mapping.get(template_type, "individual_analysis")
        import asyncio
        template = asyncio.run(self.get_template(template_name))
        return template.source if hasattr(template, 'source') else str(template)
    
    def register_custom_template(self, name: str, content: str, base_template: ReportTemplate) -> bool:
        """Register a custom template"""
        try:
            import asyncio
            asyncio.run(self.add_custom_template(name, content))
            return True
        except Exception as e:
            logger.error(f"Failed to register custom template {name}: {e}")
            return False
    
    async def add_custom_template(self, template_name: str, template_content: str, 
                                format_type: ReportFormat) -> bool:
        """Add custom template with validation"""
        
        try:
            # Create template file
            template_path = self.template_dir / f"{template_name}.html"
            
            # Write template content
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            # Clear cache to force reload
            if template_name in self._template_cache:
                del self._template_cache[template_name]
            
            # Validate template
            schema = TEMPLATE_SCHEMAS.get('analysis_report', {})
            await self.validate_template(template_name, schema)
            
            logger.info(f"Custom template added successfully: {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom template {template_name}: {e}")
            # Clean up failed template file
            template_path = self.template_dir / f"{template_name}.html"
            if template_path.exists():
                template_path.unlink()
            
            raise TemplateValidationError(f"Failed to add custom template: {e}")
    
    def _create_default_templates(self):
        """Create default templates if they don't exist"""
        
        default_templates = {
            "individual_analysis": self._get_individual_analysis_template(),
            "comparative_analysis": self._get_comparative_analysis_template(),
            "executive_summary": self._get_executive_summary_template(),
            "technical_report": self._get_technical_report_template(),
            "bulk_summary": self._get_bulk_summary_template()
        }
        
        for template_name, content in default_templates.items():
            template_path = self.template_dir / f"{template_name}.html"
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Created default template: {template_name}")
    
    async def _get_default_template(self, template_name: str) -> Template:
        """Get default template for unknown templates"""
        
        # Use individual analysis as default fallback
        fallback_content = self._get_individual_analysis_template()
        return self.env.from_string(fallback_content)
    
    def _generate_test_data(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data for template validation"""
        
        from datetime import datetime
        
        return {
            'metadata': {
                'report_id': 'test-report-123',
                'generated_at': datetime.now().isoformat(),
                'generator_version': '2.3.0',
                'template_used': 'test',
                'format_type': 'html',
                'generation_time_ms': 150.0
            },
            'url': 'https://example.com',
            'site_name': 'Example Website',
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_score': 7.5,
            'dimension_scores': {
                'content_quality': 8.0,
                'seo_optimization': 7.0,
                'user_experience': 7.5,
                'accessibility': 6.5,
                'performance': 8.0,
                'security': 7.5
            },
            'executive_summary': {
                'summary_text': 'Test executive summary for template validation',
                'key_metrics': {'score': '7.5/10', 'rank': '2nd'},
                'top_strengths': ['Good content quality', 'Strong performance'],
                'critical_issues': ['SEO optimization needed'],
                'priority_actions': ['Improve meta tags', 'Add structured data'],
                'overall_assessment': 'Good foundation with room for improvement'
            },
            'detailed_analysis': {
                'content_analysis': 'Detailed content analysis results',
                'seo_analysis': 'SEO optimization findings',
                'ux_analysis': 'User experience evaluation'
            },
            'recommendations': ['Recommendation 1', 'Recommendation 2'],
            'improvement_roadmap': [
                {
                    'category': 'seo',
                    'title': 'Improve Meta Tags',
                    'description': 'Add missing meta descriptions',
                    'priority': 'high',
                    'effort_level': 'low',
                    'expected_impact': 'medium',
                    'timeline_weeks': 2,
                    'dependencies': [],
                    'success_metrics': ['CTR improvement']
                }
            ],
            'technical_details': {'analysis_method': 'automated'},
            'appendices': {'data_sources': 'Web scraping and LLM analysis'},
            
            # Template helper functions
            'format_score': lambda score, max_score=10: f"{score:.1f}/{max_score}",
            'format_percentage': lambda value: f"{value:.1f}%",
            'format_timestamp': lambda ts: datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M UTC") if isinstance(ts, str) else ts.strftime("%Y-%m-%d %H:%M UTC"),
            'format_list': lambda items, max_items=5: ", ".join(items[:max_items]) + (f" (and {len(items) - max_items} more)" if len(items) > max_items else ""),
            'get_priority_color': lambda priority: {'critical': '#FF4444', 'high': '#FF8800', 'medium': '#FFBB00', 'low': '#00AA00'}.get(priority.lower(), '#666666'),
            'get_score_level': lambda score: 'Excellent' if score >= 8.5 else 'Good' if score >= 7.0 else 'Fair' if score >= 5.5 else 'Poor' if score >= 4.0 else 'Critical'
        }
    
    def _get_individual_analysis_template(self) -> str:
        """Get individual analysis report template"""
        
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Website Analysis Report - {{ site_name }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 3px solid #2E7D32; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #2E7D32; margin: 0; font-size: 2.5em; }
        .header .meta { color: #666; margin-top: 10px; }
        .score-card { background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0; }
        .score-card .score { font-size: 3em; font-weight: bold; margin: 0; }
        .score-card .label { font-size: 1.2em; opacity: 0.9; }
        .section { margin: 30px 0; }
        .section h2 { color: #2E7D32; border-bottom: 2px solid #E8F5E8; padding-bottom: 10px; }
        .dimension-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .dimension-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }
        .dimension-card h3 { margin-top: 0; color: #2E7D32; }
        .score-bar { background: #E0E0E0; height: 10px; border-radius: 5px; overflow: hidden; margin: 10px 0; }
        .score-fill { background: linear-gradient(90deg, #FF5722, #FF9800, #FFC107, #8BC34A, #4CAF50); height: 100%; transition: width 0.3s ease; }
        .executive-summary { background: #E8F5E8; padding: 20px; border-radius: 8px; border-left: 4px solid #2E7D32; }
        .recommendations { background: #FFF3E0; padding: 20px; border-radius: 8px; border-left: 4px solid #FF9800; }
        .roadmap-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; background: white; }
        .priority-high { border-left: 4px solid #FF5722; }
        .priority-medium { border-left: 4px solid #FF9800; }
        .priority-low { border-left: 4px solid #4CAF50; }
        .meta-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 0.9em; color: #666; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Website Analysis Report</h1>
            <div class="meta">
                <strong>{{ site_name }}</strong> | {{ url }}<br>
                Generated on {{ format_timestamp(analysis_timestamp) }} | Report ID: {{ metadata.report_id }}
            </div>
        </div>

        <!-- Overall Score -->
        <div class="score-card">
            <div class="score">{{ format_score(overall_score) }}</div>
            <div class="label">Overall Performance Score</div>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="executive-summary">
                <p>{{ executive_summary.summary_text }}</p>
                <h4>Key Metrics:</h4>
                <ul>
                    {% for key, value in executive_summary.key_metrics.items() %}
                    <li><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <!-- Dimension Scores -->
        <div class="section">
            <h2>Performance by Dimension</h2>
            <div class="dimension-grid">
                {% for dimension, score in dimension_scores.items() %}
                <div class="dimension-card">
                    <h3>{{ dimension.replace('_', ' ').title() }}</h3>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {{ score * 10 }}%"></div>
                    </div>
                    <p><strong>{{ format_score(score) }}</strong> - {{ get_score_level(score) }}</p>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Key Strengths & Issues -->
        <div class="section">
            <h2>Key Findings</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h3 style="color: #4CAF50;">Top Strengths</h3>
                    <ul>
                        {% for strength in executive_summary.top_strengths %}
                        <li>{{ strength }}</li>
                        {% endfor %}
                    </ul>
                </div>
                <div>
                    <h3 style="color: #FF5722;">Critical Issues</h3>
                    <ul>
                        {% for issue in executive_summary.critical_issues %}
                        <li>{{ issue }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>

        <!-- Recommendations -->
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                {% for recommendation in recommendations %}
                <p>• {{ recommendation }}</p>
                {% endfor %}
            </div>
        </div>

        <!-- Improvement Roadmap -->
        <div class="section">
            <h2>Improvement Roadmap</h2>
            {% for item in improvement_roadmap %}
            <div class="roadmap-item priority-{{ item.priority }}">
                <h4>{{ item.title }}</h4>
                <p>{{ item.description }}</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; font-size: 0.9em;">
                    <div><strong>Priority:</strong> {{ item.priority.title() }}</div>
                    <div><strong>Effort:</strong> {{ item.effort_level.title() }}</div>
                    <div><strong>Impact:</strong> {{ item.expected_impact.title() }}</div>
                    <div><strong>Timeline:</strong> {{ item.timeline_weeks }} weeks</div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Report Metadata -->
        <div class="meta-info">
            <strong>Report Details:</strong> Generated by Web Content Analyzer v{{ metadata.generator_version }} | 
            Template: {{ metadata.template_used }} | 
            Generation Time: {{ metadata.generation_time_ms }}ms |
            Format: {{ metadata.format_type.upper() }}
        </div>

        <div class="footer">
            <p>This report was generated automatically using advanced web content analysis techniques.</p>
        </div>
    </div>
</body>
</html>'''
    
    def _get_comparative_analysis_template(self) -> str:
        """Get comparative analysis report template"""
        
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Website Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 3px solid #1976D2; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #1976D2; margin: 0; font-size: 2.5em; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .summary-card { background: linear-gradient(135deg, #1976D2, #42A5F5); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card .value { font-size: 2.5em; font-weight: bold; margin: 0; }
        .summary-card .label { font-size: 1.1em; opacity: 0.9; }
        .section { margin: 30px 0; }
        .section h2 { color: #1976D2; border-bottom: 2px solid #E3F2FD; padding-bottom: 10px; }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .comparison-table th, .comparison-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .comparison-table th { background: #E3F2FD; font-weight: bold; color: #1976D2; }
        .comparison-table tr:nth-child(even) { background: #f9f9f9; }
        .rank-1 { background: #E8F5E8 !important; border-left: 4px solid #4CAF50; }
        .rank-2 { background: #FFF3E0 !important; border-left: 4px solid #FF9800; }
        .rank-3 { background: #FFEBEE !important; border-left: 4px solid #FF5722; }
        .differentiator { background: #F3E5F5; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #9C27B0; }
        .insight-card { background: #E8F5E8; padding: 20px; border-radius: 8px; margin: 15px 0; }
        .website-card { border: 1px solid #ddd; margin: 15px 0; padding: 20px; border-radius: 8px; background: #fafafa; }
        .website-card h3 { color: #1976D2; margin-top: 0; }
        .score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }
        .score-item { text-align: center; padding: 10px; background: white; border-radius: 4px; border: 1px solid #ddd; }
        .meta-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Comparative Website Analysis</h1>
            <div class="meta">
                {{ comparison_summary }}<br>
                Generated on {{ format_timestamp(metadata.generated_at) }} | Report ID: {{ metadata.report_id }}
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{{ websites_analyzed }}</div>
                <div class="label">Websites Analyzed</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ key_differentiators|length }}</div>
                <div class="label">Key Differentiators</div>
            </div>
            <div class="summary-card">
                <div class="value">{{ comparison_criteria|length }}</div>
                <div class="label">Analysis Dimensions</div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="insight-card">
                <p>{{ executive_summary.summary_text }}</p>
                <h4>Overall Assessment:</h4>
                <p>{{ executive_summary.overall_assessment }}</p>
            </div>
        </div>

        <!-- Overall Rankings -->
        <div class="section">
            <h2>Overall Performance Rankings</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Website</th>
                        <th>Overall Score</th>
                        <th>Performance Level</th>
                    </tr>
                </thead>
                <tbody>
                    {% for ranking in overall_rankings %}
                    <tr class="rank-{{ loop.index if loop.index <= 3 else 'other' }}">
                        <td><strong>{{ loop.index }}</strong></td>
                        <td>{{ ranking.url }}</td>
                        <td>{{ format_score(ranking.score) }}</td>
                        <td>{{ get_score_level(ranking.score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Key Differentiators -->
        <div class="section">
            <h2>Key Differentiators</h2>
            {% for differentiator in key_differentiators %}
            <div class="differentiator">
                <h4>{{ differentiator.title }}</h4>
                <p>{{ differentiator.description }}</p>
                <p><strong>Significance Score:</strong> {{ differentiator.significance_score }}/10</p>
                {% if differentiator.supporting_data %}
                <p><strong>Supporting Data:</strong>
                    {% for key, value in differentiator.supporting_data.items() %}
                    {{ key.replace('_', ' ').title() }}: {{ value }}{% if not loop.last %} | {% endif %}
                    {% endfor %}
                </p>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <!-- Website Comparisons -->
        <div class="section">
            <h2>Detailed Website Comparisons</h2>
            {% for comparison in website_comparisons %}
            <div class="website-card">
                <h3>{{ comparison.site_name }}</h3>
                <p><strong>URL:</strong> {{ comparison.url }}</p>
                <p><strong>Rank:</strong> #{{ comparison.rank_position }} | <strong>Overall Score:</strong> {{ format_score(comparison.overall_score) }}</p>
                
                <h4>Dimension Scores:</h4>
                <div class="score-grid">
                    {% for dimension, score in comparison.dimension_scores.items() %}
                    <div class="score-item">
                        <div>{{ dimension.replace('_', ' ').title() }}</div>
                        <div><strong>{{ format_score(score) }}</strong></div>
                    </div>
                    {% endfor %}
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h5>Strengths:</h5>
                        <ul>
                            {% for strength in comparison.strengths %}
                            <li>{{ strength }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    <div>
                        <h5>Weaknesses:</h5>
                        <ul>
                            {% for weakness in comparison.weaknesses %}
                            <li>{{ weakness }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    <div>
                        <h5>Differentiators:</h5>
                        <ul>
                            {% for diff in comparison.differentiators %}
                            <li>{{ diff }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Cross-Site Recommendations -->
        <div class="section">
            <h2>Cross-Site Recommendations</h2>
            {% for recommendation in cross_site_recommendations %}
            <div class="insight-card">
                <h4>{{ recommendation.title }}</h4>
                <p>{{ recommendation.description }}</p>
                <p><strong>Impact:</strong> {{ recommendation.impact }} | 
                   <strong>Timeline:</strong> {{ recommendation.timeline }} | 
                   <strong>Priority:</strong> {{ recommendation.priority }}</p>
            </div>
            {% endfor %}
        </div>

        <!-- Report Metadata -->
        <div class="meta-info">
            <strong>Report Details:</strong> Generated by Web Content Analyzer v{{ metadata.generator_version }} | 
            Template: {{ metadata.template_used }} | 
            Generation Time: {{ metadata.generation_time_ms }}ms |
            Format: {{ metadata.format_type.upper() }}
        </div>
    </div>
</body>
</html>'''
    
    def _get_executive_summary_template(self) -> str:
        """Get executive summary template"""
        
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary - {{ site_name if site_name is defined else 'Website Analysis' }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 3px solid #6A1B9A; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #6A1B9A; margin: 0; font-size: 2.2em; }
        .summary-box { background: #F3E5F5; padding: 25px; border-radius: 8px; border-left: 5px solid #6A1B9A; margin: 20px 0; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: #E8EAF6; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #9C27B0; }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #6A1B9A; margin: 0; }
        .metric-label { font-size: 0.9em; color: #666; }
        .section { margin: 25px 0; }
        .section h3 { color: #6A1B9A; border-bottom: 2px solid #F3E5F5; padding-bottom: 8px; }
        .list-item { background: #FAFAFA; padding: 10px; margin: 8px 0; border-radius: 4px; border-left: 3px solid #9C27B0; }
        .assessment-box { background: linear-gradient(135deg, #6A1B9A, #9C27B0); color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Executive Summary</h1>
            <p>{{ site_name if site_name is defined else 'Website Analysis Report' }}</p>
        </div>

        <div class="summary-box">
            <p>{{ executive_summary.summary_text }}</p>
        </div>

        <div class="section">
            <h3>Key Performance Metrics</h3>
            <div class="metrics-grid">
                {% for key, value in executive_summary.key_metrics.items() %}
                <div class="metric-card">
                    <div class="metric-value">{{ value }}</div>
                    <div class="metric-label">{{ key.replace('_', ' ').title() }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h3>Top Strengths</h3>
            {% for strength in executive_summary.top_strengths %}
            <div class="list-item">{{ strength }}</div>
            {% endfor %}
        </div>

        <div class="section">
            <h3>Critical Issues</h3>
            {% for issue in executive_summary.critical_issues %}
            <div class="list-item" style="border-left-color: #FF5722; background: #FFEBEE;">{{ issue }}</div>
            {% endfor %}
        </div>

        <div class="section">
            <h3>Priority Actions</h3>
            {% for action in executive_summary.priority_actions %}
            <div class="list-item" style="border-left-color: #FF9800; background: #FFF3E0;">{{ action }}</div>
            {% endfor %}
        </div>

        <div class="assessment-box">
            <h3 style="margin-top: 0; color: white;">Overall Assessment</h3>
            <p style="margin-bottom: 0;">{{ executive_summary.overall_assessment }}</p>
        </div>
    </div>
</body>
</html>'''
    
    def _get_technical_report_template(self) -> str:
        """Get technical report template"""
        
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Analysis Report - {{ site_name }}</title>
    <style>
        body { font-family: 'Courier New', monospace; line-height: 1.4; margin: 0; padding: 20px; background: #2E2E2E; color: #E0E0E0; }
        .container { max-width: 1200px; margin: 0 auto; background: #1E1E1E; padding: 30px; border-radius: 8px; border: 1px solid #444; }
        .header { border-bottom: 2px solid #00BCD4; padding-bottom: 15px; margin-bottom: 25px; }
        .header h1 { color: #00BCD4; margin: 0; font-size: 2em; }
        .code-block { background: #000; padding: 15px; border-radius: 4px; border-left: 4px solid #00BCD4; margin: 15px 0; overflow-x: auto; }
        .technical-section { background: #2A2A2A; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #444; }
        .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #444; }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #00BCD4; }
        .metric-value { color: #4CAF50; font-weight: bold; }
        .data-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .data-table th, .data-table td { border: 1px solid #444; padding: 8px; text-align: left; }
        .data-table th { background: #333; color: #00BCD4; }
        .warning { color: #FF9800; }
        .error { color: #FF5722; }
        .success { color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Technical Analysis Report</h1>
            <div>{{ site_name }} | {{ url }}</div>
            <div>Generated: {{ format_timestamp(analysis_timestamp) }}</div>
        </div>

        <div class="technical-section">
            <h3>Performance Metrics</h3>
            {% for dimension, score in dimension_scores.items() %}
            <div class="metric-row">
                <span class="metric-label">{{ dimension.replace('_', ' ').upper() }}:</span>
                <span class="metric-value">{{ format_score(score) }}</span>
            </div>
            {% endfor %}
        </div>

        <div class="technical-section">
            <h3>Technical Details</h3>
            <div class="code-block">
{{ technical_details | tojson(indent=2) }}
            </div>
        </div>

        <div class="technical-section">
            <h3>Analysis Data</h3>
            <div class="code-block">
Report ID: {{ metadata.report_id }}
Generator Version: {{ metadata.generator_version }}
Generation Time: {{ metadata.generation_time_ms }}ms
Template: {{ metadata.template_used }}
Format: {{ metadata.format_type }}
            </div>
        </div>

        {% if improvement_roadmap %}
        <div class="technical-section">
            <h3>Implementation Roadmap</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Priority</th>
                        <th>Task</th>
                        <th>Effort</th>
                        <th>Impact</th>
                        <th>Timeline</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in improvement_roadmap %}
                    <tr>
                        <td class="{{ 'error' if item.priority == 'critical' else 'warning' if item.priority == 'high' else 'success' }}">
                            {{ item.priority.upper() }}
                        </td>
                        <td>{{ item.title }}</td>
                        <td>{{ item.effort_level.upper() }}</td>
                        <td>{{ item.expected_impact.upper() }}</td>
                        <td>{{ item.timeline_weeks }}w</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="technical-section">
            <h3>Raw Data Export</h3>
            <div class="code-block">
# Analysis Results Export
# Generated: {{ format_timestamp(metadata.generated_at) }}
# 
# Overall Score: {{ overall_score }}/10
# Dimensions Analyzed: {{ dimension_scores.keys() | list | length }}
# Recommendations: {{ recommendations | length }}
# 
# END REPORT
            </div>
        </div>
    </div>
</body>
</html>'''
    
    def _get_bulk_summary_template(self) -> str:
        """Get bulk summary template"""
        
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bulk Analysis Summary Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 3px solid #FF5722; padding-bottom: 20px; margin-bottom: 30px; text-align: center; }
        .header h1 { color: #FF5722; margin: 0; font-size: 2.5em; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: linear-gradient(135deg, #FF5722, #FF7043); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2.5em; font-weight: bold; margin: 0; }
        .stat-label { font-size: 1.1em; opacity: 0.9; }
        .section { margin: 30px 0; }
        .section h2 { color: #FF5722; border-bottom: 2px solid #FFEBEE; padding-bottom: 10px; }
        .performance-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .performance-table th, .performance-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .performance-table th { background: #FFEBEE; color: #FF5722; font-weight: bold; }
        .insight-box { background: #FFF3E0; padding: 20px; border-radius: 8px; border-left: 4px solid #FF9800; margin: 15px 0; }
        .issue-item { background: #FFEBEE; padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 3px solid #FF5722; }
        .success-rate { background: #E8F5E8; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
        .meta-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Bulk Analysis Summary</h1>
            <p>Comprehensive analysis of {{ total_urls_analyzed }} websites</p>
            <p>Generated on {{ format_timestamp(metadata.generated_at) }}</p>
        </div>

        <!-- Summary Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ total_reports_generated }}</div>
                <div class="stat-label">Total Reports</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ successful_reports }}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ failed_reports }}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_urls_analyzed }}</div>
                <div class="stat-label">URLs Analyzed</div>
            </div>
        </div>

        <!-- Success Rate -->
        <div class="section">
            <h2>Processing Summary</h2>
            <div class="success-rate">
                <h4>Success Rate: {{ performance_metrics.success_rate }}</h4>
                <p>Average Generation Time: {{ performance_metrics.avg_time_per_report }}</p>
                <p>Total Processing Time: {{ performance_metrics.total_processing_time }}</p>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="section">
            <h2>Performance Metrics</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Reports Generated</td>
                        <td>{{ total_reports_generated }}</td>
                        <td>{{ 'Success' if successful_reports == total_reports_generated else 'Partial' }}</td>
                    </tr>
                    <tr>
                        <td>Success Rate</td>
                        <td>{{ performance_metrics.success_rate }}</td>
                        <td>{{ 'Excellent' if performance_metrics.success_rate|float >= 95 else 'Good' if performance_metrics.success_rate|float >= 85 else 'Needs Improvement' }}</td>
                    </tr>
                    <tr>
                        <td>Average Time per Report</td>
                        <td>{{ performance_metrics.avg_time_per_report }}</td>
                        <td>{{ 'Fast' if average_generation_time_ms < 1000 else 'Normal' if average_generation_time_ms < 3000 else 'Slow' }}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Common Issues -->
        <div class="section">
            <h2>Common Issues Identified</h2>
            {% for issue in common_issues_identified %}
            <div class="issue-item">{{ issue }}</div>
            {% endfor %}
        </div>

        <!-- Bulk Insights -->
        <div class="section">
            <h2>Key Insights</h2>
            {% for insight in bulk_insights %}
            <div class="insight-box">{{ insight }}</div>
            {% endfor %}
        </div>

        <div class="meta-info">
            <strong>Report Details:</strong> Generated by Web Content Analyzer v{{ metadata.generator_version }} | 
            Template: {{ metadata.template_used }} | 
            Total Generation Time: {{ metadata.generation_time_ms }}ms |
            Format: {{ metadata.format_type.upper() }}
        </div>
    </div>
</body>
</html>'''
