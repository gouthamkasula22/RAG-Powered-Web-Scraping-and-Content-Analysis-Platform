# Web Content Analyzer - Clean Architecture

## 📁 Project Structure

```
Web Content Analysis/
├── 📁 backend/                     # Backend API Services
│   ├── 📁 api/                     # FastAPI application
│   │   └── main.py                 # FastAPI app with CORS
│   ├── 📁 src/                     # Core business logic
│   │   ├── 📁 application/         # Application layer
│   │   ├── 📁 domain/              # Domain models
│   │   ├── 📁 infrastructure/      # Infrastructure layer
│   │   └── 📁 presentation/        # Presentation layer
│   └── launch_backend.py           # Backend launcher
├── 📁 frontend/                    # Frontend Applications
│   └── 📁 streamlit/               # Streamlit web interface
│       ├── 📁 .streamlit/          # Streamlit configuration
│       ├── app.py                  # Main Streamlit app
│       └── launch_frontend.py      # Frontend launcher
├── 📁 scripts/                     # Development scripts
├── 📁 tests/                       # Test suites
├── launch_full_stack.py            # Complete system launcher
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Option 1: Full Stack (Recommended)
```bash
python launch_full_stack.py
# Choose option 1 for full stack
```

### Option 2: Individual Services

**Backend Only:**
```bash
cd backend
python launch_backend.py
# API available at: http://localhost:8000
```

**Frontend Only:**
```bash
cd frontend/streamlit
python launch_frontend.py
# Interface available at: http://localhost:8501
```

## 🔧 Features

### Backend (FastAPI)
- ✅ **CORS Middleware** - Properly configured for cross-origin requests
- ✅ **Professional API** - RESTful endpoints with proper error handling
- ✅ **Health Monitoring** - Service health checks and provider status
- ✅ **Request/Response Models** - Type-safe Pydantic models
- ✅ **Auto Documentation** - OpenAPI/Swagger docs at `/api/docs`

### Frontend (Streamlit)
- ✅ **Professional UI** - Clean, minimal business dashboard
- ✅ **Real-time Analysis** - Progress tracking and live updates
- ✅ **Interactive Results** - Tabbed content, charts, export options
- ✅ **Strategic Emojis** - Minimal, purposeful emoji usage
- ✅ **Export Functionality** - JSON export with PDF/CSV coming soon

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with service info
- `GET /api/health` - Health check with provider status
- `POST /api/analyze` - Analyze website content
- `GET /api/docs` - Interactive API documentation

### Example API Usage

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Content Analysis:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
-H "Content-Type: application/json" \
-d '{
  "url": "https://example.com",
  "analysis_type": "comprehensive",
  "quality_preference": "balanced",
  "max_cost": 0.05
}'
```

## 🛡️ CORS Configuration

The backend includes comprehensive CORS configuration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:8501",  # Streamlit
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)
```

## 🔧 Development

### Backend Development
```bash
cd backend/api
uvicorn main:app --reload --port 8000
```

### Frontend Development
```bash
cd frontend/streamlit
streamlit run app.py --server.port 8501
```

## 📦 Dependencies

Core packages are already in `requirements.txt`:
- **FastAPI** - Modern web framework
- **Streamlit** - Web interface framework
- **Plotly** - Interactive charts
- **Pandas** - Data manipulation
- **Uvicorn** - ASGI server

## 🏗️ Architecture Benefits

### Clean Separation
- **Backend**: Pure API with business logic
- **Frontend**: UI presentation layer
- **Shared**: Domain models and interfaces

### Scalability
- Services can be deployed independently
- Frontend can connect to remote backend
- Multiple frontend options possible

### Development
- Teams can work on frontend/backend separately
- Clear API contracts
- Easy testing and debugging

## 🎯 Next Steps

1. **Backend Enhancement**
   - Add authentication/authorization
   - Implement analysis history storage
   - Add rate limiting and caching

2. **Frontend Enhancement**
   - Add real-time backend connectivity
   - Implement PDF/CSV export
   - Add analysis history UI

3. **Deployment**
   - Docker containerization
   - Production configuration
   - CI/CD pipeline setup

## 🔍 Testing

**Backend Health:**
```bash
curl http://localhost:8000/api/health
```

**Frontend Access:**
Visit: http://localhost:8501

**Full API Documentation:**
Visit: http://localhost:8000/api/docs
