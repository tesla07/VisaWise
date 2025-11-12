# VisaWise Quick Start Guide

## Overview
VisaWise is an AI-powered system for USCIS query resolution with multiple agent types and comprehensive monitoring.

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- OpenAI API key

### 2. Installation

#### Option A: Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/tesla07/VisaWise.git
cd VisaWise

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f visawise-api
```

#### Option B: Local Development
```bash
# Clone the repository
git clone https://github.com/tesla07/VisaWise.git
cd VisaWise

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the application
python main.py
```

### 3. Accessing Services

Once running, access:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Using the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Query with LangGraph Agent
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does RFE mean for my USCIS case?",
    "agent_type": "langgraph",
    "session_id": "my-session-123"
  }'
```

### Query with LangChain Agent
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the H-1B visa process",
    "agent_type": "langchain"
  }'
```

### Query with CrewAI Agents
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Check status of case WAC2190012345 and explain what to do next",
    "agent_type": "crewai"
  }'
```

### Check Case Status Directly
```bash
curl -X POST http://localhost:8000/case-status \
  -H "Content-Type: application/json" \
  -d '{
    "receipt_number": "WAC2190012345"
  }'
```

### Get MCP Tools
```bash
curl http://localhost:8000/mcp/tools
```

### Get Session Context
```bash
curl http://localhost:8000/mcp/context/my-session-123
```

### View Metrics
```bash
curl http://localhost:8000/metrics
```

## Agent Types Comparison

### LangChain Agent
- **Use for**: Direct queries requiring specific tools
- **Strengths**: Fast, structured function calling
- **Best for**: Simple case status checks, single-topic questions

### LangGraph Workflow
- **Use for**: Complex queries requiring multi-step processing
- **Strengths**: Stateful, conditional routing, context preservation
- **Best for**: Queries that need clarification, multi-step processes

### CrewAI Agents
- **Use for**: Comprehensive questions requiring multiple expertise
- **Strengths**: Collaborative specialists working together
- **Best for**: Complex immigration questions, holistic guidance

## Example Usage Script

Run the included example script:
```bash
python examples/example_usage.py
```

This will demonstrate:
1. Health check
2. LangGraph query
3. Direct case status check
4. CrewAI collaborative query
5. MCP tools listing
6. Session context retrieval
7. Metrics inspection

## Monitoring with Grafana

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards
4. Select "VisaWise Monitoring Dashboard"

You'll see:
- Query rate by agent type
- Query duration percentiles
- Case status check rate
- Active sessions count
- Error rates
- Cache size

## Testing

Run the test suite:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_uscis_service.py -v

# With coverage
pip install pytest-cov
pytest --cov=src/visawise --cov-report=html
```

## Configuration

Edit `.env` file:
```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional - defaults work for local development
API_HOST=0.0.0.0
API_PORT=8000
REDIS_HOST=localhost
REDIS_PORT=6379
GRAFANA_HOST=localhost
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

## Common Use Cases

### 1. Check Case Status
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/case-status",
        json={"receipt_number": "WAC2190012345"}
    )
    print(response.json())
```

### 2. Get Immigration Information
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/query",
        json={
            "query": "What are the requirements for an H-1B visa?",
            "agent_type": "langchain"
        }
    )
    print(response.json()["response"])
```

### 3. Track Session Context
```python
import httpx

session_id = "user-123"

async with httpx.AsyncClient() as client:
    # Make queries with session tracking
    await client.post(
        "http://localhost:8000/query",
        json={
            "query": "Check status WAC2190012345",
            "agent_type": "langgraph",
            "session_id": session_id
        }
    )
    
    # Retrieve session context
    response = await client.get(
        f"http://localhost:8000/mcp/context/{session_id}"
    )
    print(response.json())
```

## Troubleshooting

### API Not Starting
- Check if port 8000 is available
- Verify OPENAI_API_KEY is set in .env
- Check logs: `docker-compose logs visawise-api`

### No Response from Agents
- Verify OpenAI API key is valid
- Check API rate limits
- Review logs for errors

### Grafana Dashboard Empty
- Ensure Prometheus is running
- Verify metrics endpoint: http://localhost:8000/metrics
- Check Grafana datasource configuration

### Tests Failing
- Install all test dependencies: `pip install -r requirements.txt`
- Ensure .env file exists
- Run with verbose output: `pytest -v`

## Architecture Overview

```
User Request
     ↓
FastAPI Endpoint
     ↓
Agent Selection (LangChain/LangGraph/CrewAI)
     ↓
USCIS Service (if needed)
     ↓
Response Generation
     ↓
Metrics Recording
     ↓
Response to User
```

## Development

### Adding New Endpoints
1. Edit `src/visawise/api/app.py`
2. Add route handler
3. Update tests in `tests/test_api.py`

### Adding New Agents
1. Create agent class in `src/visawise/agents/`
2. Register in `src/visawise/agents/__init__.py`
3. Update API to support new agent type

### Custom Metrics
1. Add metric in `src/visawise/utils/monitoring.py`
2. Record metric in appropriate location
3. Add to Grafana dashboard

## Production Deployment

### Using Docker Compose
```bash
# Production mode
docker-compose -f docker-compose.yml up -d

# Scale API service
docker-compose up -d --scale visawise-api=3
```

### Environment Variables for Production
```bash
OPENAI_API_KEY=prod_key
API_HOST=0.0.0.0
API_PORT=8000
REDIS_HOST=redis-prod
LOG_LEVEL=WARNING
```

### Monitoring in Production
- Set up Grafana alerts
- Monitor error rates
- Track response times
- Set up log aggregation

## Support

- GitHub Issues: https://github.com/tesla07/VisaWise/issues
- Documentation: See README.md
- Examples: See examples/ directory

## License

MIT License - See LICENSE file for details
