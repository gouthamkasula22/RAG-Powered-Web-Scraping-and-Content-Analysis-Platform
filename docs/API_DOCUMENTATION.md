# API Documentation

## Web Content Analysis REST API v1.0

The Web Content Analysis API provides programmatic access to AI-powered website content analysis capabilities. This RESTful API enables developers to integrate intelligent website analysis into their applications.

## Base URL

```
http://localhost:8000/api
```

**Production URL**: To be configured based on deployment environment.

## Authentication

The API currently operates in development mode without authentication. Production deployments implement JWT-based authentication for secure access.

**Future Authentication Header:**
```
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

### System Health

#### GET /health

Retrieve the current health status of the API and all integrated services.

**Response Example:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "llm_service": "operational",
    "analysis_service": "operational", 
    "scraping_service": "operational"
  },
  "providers": {
    "gemini": {
      "available": true,
      "cost_per_1k_tokens": 0.0,
      "max_tokens": 1048576
    },
    "claude": {
      "available": true,
      "cost_per_1k_tokens": 0.25,
      "max_tokens": 200000
    }
  }
}
```

### Single Website Analysis

#### POST /analyze

Analyze a single website's content with comprehensive AI-powered insights.

**Request Parameters:**
- `url` (string, required): The website URL to analyze
- `analysis_type` (string, optional): Analysis depth - "basic", "comprehensive", "detailed"
- `quality_preference` (string, optional): Processing quality - "fast", "balanced", "high"
- `max_cost` (float, optional): Maximum cost threshold in USD

**Request Example:**
```json
{
  "url": "https://example.com",
  "analysis_type": "comprehensive",
  "quality_preference": "balanced",
  "max_cost": 0.05
}
```

**Response Example:**
```json
{
  "analysis_id": "abc12345",
  "url": "https://example.com",
  "status": "completed",
  "executive_summary": "The website demonstrates strong content organization with clear navigation structure. The main content areas are well-defined, though there are opportunities for SEO enhancement through improved meta descriptions and structured data implementation.",
  "metrics": {
    "overall_score": 7.5,
    "content_quality_score": 8.0,
    "seo_score": 7.0,
    "ux_score": 7.5,
    "readability_score": 8.0,
    "engagement_score": 7.0
  },
  "insights": {
    "strengths": [
      "Clear navigation structure",
      "Mobile responsive design",
      "Fast loading performance"
    ],
    "weaknesses": [
      "Missing meta descriptions on key pages",
      "Limited structured data markup",
      "Inconsistent heading hierarchy"
    ],
    "opportunities": [
      "Implement schema.org markup",
      "Optimize images with alt text",
      "Add internal linking strategy"
    ],
    "recommendations": [
      "Implement page-specific meta descriptions",
      "Add structured data for better search visibility",
      "Create content hub architecture"
    ],
    "key_findings": [
      "Content quality is above average",
      "SEO foundation is solid but incomplete",
      "User experience meets modern standards"
    ]
  },
  "scraped_content": {
    "title": "Example Website - Leading Solutions Provider",
    "main_content": "Welcome to our comprehensive platform...",
    "meta_description": "Discover innovative solutions for your business needs",
    "headings": ["Welcome", "Our Services", "About Us", "Contact"],
    "url": "https://example.com",
    "word_count": 450
  },
  "processing_time": 2.3,
  "cost": 0.001,
  "provider_used": "gemini",
  "created_at": "2024-01-15T10:30:00Z",
  "error_message": null
}
```

### Bulk Website Analysis

#### POST /v1/analyze/bulk

Analyze multiple websites simultaneously with parallel processing capabilities.

**Request Parameters:**
- `urls` (array, required): List of website URLs to analyze
- `analysis_type` (string, optional): Analysis depth for all websites
- `quality_preference` (string, optional): Processing quality preference
- `max_cost` (float, optional): Maximum cost threshold for entire batch
- `parallel_limit` (integer, optional): Number of concurrent analyses (maximum 5)

**Request Example:**
```json
{
  "urls": [
    "https://example.com",
    "https://competitor.com",
    "https://industry-leader.com"
  ],
  "analysis_type": "comprehensive",
  "quality_preference": "balanced",
  "max_cost": 0.15,
  "parallel_limit": 3
}
```

**Response Example:**
```json
{
  "batch_id": "batch_xyz789",
  "total_urls": 3,
  "completed": 3,
  "failed": 0,
  "results": [
    {
      "analysis_id": "abc12345",
      "url": "https://example.com",
      "status": "completed",
      "executive_summary": "Comprehensive analysis summary...",
      "metrics": {
        "overall_score": 7.5,
        "content_quality_score": 8.0,
        "seo_score": 7.0
      },
      "processing_time": 2.3,
      "cost": 0.001,
      "provider_used": "gemini",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_cost": 0.003,
  "total_processing_time": 6.8,
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:07Z"
}
```

### Knowledge Repository Operations

#### POST /knowledge-repository/add-website

Add an analyzed website to the RAG knowledge repository for future querying.

**Request Example:**
```json
{
  "website_url": "https://example.com",
  "content_chunks": [
    {
      "content": "About our company: We provide innovative solutions...",
      "metadata": {
        "section": "about",
        "word_count": 150,
        "relevance_score": 0.95
      }
    },
    {
      "content": "Our services include web development, SEO, and digital marketing...",
      "metadata": {
        "section": "services",
        "word_count": 200,
        "relevance_score": 0.92
      }
    }
  ]
}
```

**Response Example:**
```json
{
  "message": "Website added successfully to knowledge repository",
  "website_id": 42,
  "chunks_added": 2,
  "processing_time": 0.5
}
```

#### GET /knowledge-repository/websites

Retrieve a list of all websites stored in the knowledge repository.

**Query Parameters:**
- `limit` (integer, optional): Maximum number of results (default: 20)
- `offset` (integer, optional): Number of results to skip (default: 0)

**Response Example:**
```json
{
  "websites": [
    {
      "id": 1,
      "url": "https://example.com",
      "title": "Example Website - Leading Solutions Provider",
      "description": "Comprehensive business solutions and services",
      "added_at": "2024-01-15T10:30:00Z",
      "chunk_count": 5,
      "last_queried": "2024-01-15T11:45:00Z"
    },
    {
      "id": 2,
      "url": "https://competitor.com",
      "title": "Competitor Solutions",
      "description": "Alternative business solutions provider",
      "added_at": "2024-01-15T10:35:00Z",
      "chunk_count": 8,
      "last_queried": null
    }
  ],
  "total_count": 15,
  "pagination": {
    "current_page": 1,
    "total_pages": 1,
    "has_next": false,
    "has_previous": false
  }
}
```

#### POST /knowledge-repository/query

Query the knowledge repository using natural language questions.

**Request Example:**
```json
{
  "question": "What services does the company offer and what are their specializations?",
  "website_filter": "https://example.com",
  "limit": 5,
  "include_sources": true
}
```

**Response Example:**
```json
{
  "answer": "Based on the analyzed content, the company offers comprehensive web development services, SEO optimization, and digital marketing solutions. Their specializations include custom software development, e-commerce platforms, and content management systems. They have particular expertise in React and Node.js development.",
  "sources": [
    {
      "website_url": "https://example.com",
      "content_snippet": "We specialize in custom web development using modern frameworks like React and Node.js...",
      "relevance_score": 0.95,
      "section": "services",
      "word_count": 180
    },
    {
      "website_url": "https://example.com", 
      "content_snippet": "Our SEO services include keyword research, content optimization, and technical SEO audits...",
      "relevance_score": 0.88,
      "section": "seo-services",
      "word_count": 220
    }
  ],
  "processing_time": 0.8,
  "method": "AI Response (Gemini)",
  "confidence_score": 0.92
}
```

#### DELETE /knowledge-repository/website/{website_id}

Remove a specific website and all its associated content from the knowledge repository.

**Path Parameters:**
- `website_id` (integer, required): The unique identifier of the website to remove

**Response Example:**
```json
{
  "message": "Website removed successfully from knowledge repository",
  "website_id": 42,
  "deleted_chunks": 5,
  "processing_time": 0.2
}
```

#### DELETE /knowledge-repository/clear

Clear all data from the knowledge repository. This operation is irreversible.

**Response Example:**
```json
{
  "message": "Knowledge repository cleared successfully",
  "deleted_websites": 15,
  "deleted_chunks": 127,
  "processing_time": 1.5
}
```

## Error Handling

### Standard Error Response Format

All API errors follow a consistent response format:

```json
{
  "detail": "Human-readable error message",
  "error": "Detailed technical error information",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/api/analyze"
}
```

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request completed successfully |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters or format |
| 401 | Unauthorized | Authentication credentials missing or invalid |
| 403 | Forbidden | Insufficient permissions for requested operation |
| 404 | Not Found | Requested resource does not exist |
| 422 | Unprocessable Entity | Request validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | External service temporarily unavailable |

### Common Error Scenarios

**Invalid URL Format (422)**
```json
{
  "detail": "Invalid URL format provided",
  "error": "URL must start with http:// or https:// and be a valid web address",
  "status_code": 422
}
```

**Rate Limit Exceeded (429)**
```json
{
  "detail": "Rate limit exceeded",
  "error": "Maximum of 60 requests per minute allowed. Please wait before making additional requests.",
  "status_code": 429,
  "retry_after": 30
}
```

**Service Unavailable (503)**
```json
{
  "detail": "Analysis service temporarily unavailable",
  "error": "LLM providers are currently experiencing high demand. Please try again in a few moments.",
  "status_code": 503,
  "estimated_retry_time": 120
}
```

## Rate Limiting

The API implements intelligent rate limiting to ensure fair usage and optimal performance:

### Rate Limits by Endpoint Category

| Endpoint Category | Requests per Minute | Burst Limit |
|-------------------|--------------------| ------------|
| Analysis endpoints | 60 | 10 |
| Knowledge repository | 120 | 20 |
| Health check | Unlimited | N/A |

### Rate Limit Headers

All API responses include rate limiting information in the headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1642248600
X-RateLimit-Burst-Remaining: 9
```

## SDK and Integration Examples

### Python SDK Example

```python
import requests
import json
from datetime import datetime

class WebContentAnalysisClient:
    def __init__(self, base_url="http://localhost:8000/api"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def analyze_website(self, url, analysis_type="comprehensive"):
        """Analyze a single website."""
        payload = {
            "url": url,
            "analysis_type": analysis_type,
            "quality_preference": "balanced",
            "max_cost": 0.05
        }
        
        response = self.session.post(f"{self.base_url}/analyze", json=payload)
        response.raise_for_status()
        return response.json()
    
    def bulk_analyze(self, urls, parallel_limit=3):
        """Analyze multiple websites simultaneously."""
        payload = {
            "urls": urls,
            "analysis_type": "comprehensive",
            "parallel_limit": parallel_limit,
            "max_cost": len(urls) * 0.05
        }
        
        response = self.session.post(f"{self.base_url}/v1/analyze/bulk", json=payload)
        response.raise_for_status()
        return response.json()
    
    def query_knowledge_repository(self, question, website_filter=None):
        """Query the knowledge repository with natural language."""
        payload = {
            "question": question,
            "website_filter": website_filter,
            "limit": 5,
            "include_sources": True
        }
        
        response = self.session.post(f"{self.base_url}/knowledge-repository/query", json=payload)
        response.raise_for_status()
        return response.json()

# Usage example
client = WebContentAnalysisClient()

# Single website analysis
result = client.analyze_website("https://example.com")
print(f"Overall Score: {result['metrics']['overall_score']}")
print(f"Executive Summary: {result['executive_summary']}")

# Bulk analysis
websites = ["https://example.com", "https://competitor.com"]
bulk_result = client.bulk_analyze(websites)
print(f"Analyzed {bulk_result['completed']} websites in {bulk_result['total_processing_time']} seconds")

# Knowledge repository query
answer = client.query_knowledge_repository("What are the main services offered?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {len(answer['sources'])} relevant sources found")
```

### JavaScript/Node.js Example

```javascript
class WebContentAnalysisClient {
    constructor(baseUrl = 'http://localhost:8000/api') {
        this.baseUrl = baseUrl;
    }
    
    async analyzeWebsite(url, analysisType = 'comprehensive') {
        const payload = {
            url: url,
            analysis_type: analysisType,
            quality_preference: 'balanced',
            max_cost: 0.05
        };
        
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async queryKnowledgeRepository(question, websiteFilter = null) {
        const payload = {
            question: question,
            website_filter: websiteFilter,
            limit: 5,
            include_sources: true
        };
        
        const response = await fetch(`${this.baseUrl}/knowledge-repository/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage example
const client = new WebContentAnalysisClient();

// Analyze website
client.analyzeWebsite('https://example.com')
    .then(result => {
        console.log('Overall Score:', result.metrics.overall_score);
        console.log('Executive Summary:', result.executive_summary);
    })
    .catch(error => console.error('Error:', error));

// Query knowledge repository
client.queryKnowledgeRepository('What services are offered?')
    .then(answer => {
        console.log('Answer:', answer.answer);
        console.log('Sources:', answer.sources.length, 'relevant sources found');
    })
    .catch(error => console.error('Error:', error));
```

## Best Practices

### Optimal Request Patterns

1. **Batch Related Requests**: Use bulk analysis for multiple related websites
2. **Implement Retry Logic**: Handle temporary service unavailability gracefully
3. **Cache Results**: Store analysis results to avoid redundant API calls
4. **Monitor Rate Limits**: Respect rate limiting to ensure consistent access
5. **Error Handling**: Implement comprehensive error handling for all scenarios

### Performance Optimization

1. **Use Appropriate Analysis Types**: Choose "basic" for quick insights, "comprehensive" for detailed analysis
2. **Set Reasonable Timeouts**: Configure client timeouts to handle variable processing times
3. **Parallel Processing**: Utilize bulk endpoints for improved throughput
4. **Efficient Querying**: Use specific website filters in knowledge repository queries

### Security Considerations

1. **Validate URLs**: Ensure URLs are properly formatted and from trusted sources
2. **Secure API Keys**: Store API keys securely and rotate regularly
3. **Input Sanitization**: Sanitize all user inputs before sending to the API
4. **HTTPS Only**: Always use HTTPS in production environments

---

For additional support or questions about the API, please refer to our [GitHub Issues](https://github.com/yourusername/web-content-analysis/issues) or [Documentation Wiki](https://github.com/yourusername/web-content-analysis/wiki).
