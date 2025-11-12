# VisaWise

AI-powered USCIS Query Resolution System with real-time case status checks, collaborative agent workflows, and comprehensive observability.

## Overview

VisaWise is an advanced system that integrates multiple AI frameworks and tools to provide intelligent assistance for USCIS immigration queries:

- **LangChain**: LLM orchestration and tool integration
- **LangGraph**: Stateful workflow management for complex query processing
- **CrewAI**: Collaborative multi-agent systems for specialized expertise
- **OpenAI SDK**: Direct integration with OpenAI's language models
- **MCP (Model Context Protocol)**: Context sharing and tool registration
- **Grafana + Prometheus**: Real-time monitoring and observability

## Features

### 🤖 Multi-Agent Architecture
- **LangChain Agent**: Function-calling agent with tools for case checking and visa information
- **LangGraph Workflow**: Stateful workflow with conditional routing and context preservation
- **CrewAI Agents**: Collaborative specialists (Status Checker, Process Expert, Router, Synthesizer)

### 📊 Real-Time USCIS Integration
- Case status checking with receipt number validation
- Status interpretation and next-step guidance
- Visa process explanation and timeline information

### 🔍 Observability & Monitoring
- Prometheus metrics collection
- Grafana dashboards for visualization
- Request tracking and performance monitoring
- Error tracking and alerting

### 🔌 MCP Integration
- Tool registration and discovery
- Session context management
- Context export/import capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VisaWise API                            │
│                     (FastAPI)                                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  LangChain  │  │  LangGraph  │  │   CrewAI    │        │
│  │   Agent     │  │  Workflow   │  │   Agents    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                         │                                    │
│                  ┌──────▼──────┐                            │
│                  │ USCIS Service│                            │
│                  └─────────────┘                             │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐        ┌────────┐        ┌────────────┐      │
│  │   MCP    │        │ Redis  │        │ Prometheus │      │
│  │  Server  │        │ Cache  │        │  Metrics   │      │
│  └──────────┘        └────────┘        └────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ Grafana  │
                        │Dashboard │
                        └──────────┘
```

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for containerized deployment)
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tesla07/VisaWise.git
cd VisaWise
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Running with Docker Compose (Recommended)

Start all services (API, Redis, Prometheus, Grafana):
```bash
docker-compose up -d
```

Access the services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Running Locally

1. Start the API server:
```bash
python main.py
```

2. In another terminal, run the example script:
```bash
python examples/example_usage.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Query Processing
```bash
POST /query
{
  "query": "What does RFE mean for my case?",
  "agent_type": "langgraph",  # or "langchain", "crewai"
  "session_id": "optional-session-id"
}
```

### Case Status Check
```bash
POST /case-status
{
  "receipt_number": "WAC2190012345"
}
```

### MCP Tools
```bash
GET /mcp/tools
```

### Metrics
```bash
GET /metrics
```

## Agent Types

### 1. LangChain Agent
- Function-calling agent with structured tools
- Best for: Direct queries with tool usage
- Tools: check_case_status, explain_visa_process, interpret_status

### 2. LangGraph Workflow
- Stateful workflow with conditional routing
- Best for: Complex queries requiring multi-step processing
- Features: Query parsing, status checking, analysis, clarification

### 3. CrewAI Agents
- Collaborative multi-agent system
- Best for: Comprehensive answers requiring multiple expertise areas
- Agents: Status Specialist, Process Expert, Router, Synthesizer

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
pylint src/
```

## Monitoring

### Grafana Dashboard

The system includes a pre-configured Grafana dashboard with panels for:
- Query rate by agent type
- Query duration (p50, p95)
- Case status check rate
- Active sessions
- Error rate
- Cache size

Access at: http://localhost:3000

### Prometheus Metrics

Available metrics:
- `visawise_queries_total`: Total queries by agent type and status
- `visawise_query_duration_seconds`: Query processing time histogram
- `visawise_case_checks_total`: Case status checks by result
- `visawise_case_check_duration_seconds`: Case check time histogram
- `visawise_active_sessions`: Current active sessions
- `visawise_cache_size`: Cache size
- `visawise_errors_total`: Errors by type

## Configuration

Edit `.env` file to configure:
- OpenAI API key and model
- USCIS API endpoint
- Redis connection
- API server settings
- Grafana/Prometheus ports

## Project Structure

```
VisaWise/
├── src/visawise/
│   ├── agents/          # AI agents (LangChain, LangGraph, CrewAI)
│   ├── api/             # FastAPI application
│   ├── config/          # Configuration management
│   ├── mcp/             # Model Context Protocol server
│   ├── services/        # USCIS service integration
│   └── utils/           # Utilities (monitoring, etc.)
├── tests/               # Test suite
├── examples/            # Example usage scripts
├── grafana/             # Grafana configuration and dashboards
├── prometheus/          # Prometheus configuration
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Docker image definition
├── requirements.txt     # Python dependencies
└── main.py             # Application entry point
```

## Technologies

- **LangChain 0.1+**: LLM application framework
- **LangGraph 0.0.20+**: Stateful agent workflows
- **CrewAI 0.1+**: Multi-agent collaboration
- **OpenAI SDK 1.10+**: Direct OpenAI API access
- **FastAPI**: Modern async API framework
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Redis**: Caching and session management
- **Docker**: Containerization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Disclaimer

This system provides informational assistance only. Users should:
- Verify information with official USCIS resources
- Consult immigration attorneys for legal advice
- Not rely solely on AI-generated responses for critical decisions

## Support

For issues and questions:
- GitHub Issues: https://github.com/tesla07/VisaWise/issues
- Documentation: See `/docs` directory (coming soon)

## Roadmap

- [ ] Add support for more visa types
- [ ] Implement caching layer for case status
- [ ] Add WebSocket support for real-time updates
- [ ] Expand MCP tool capabilities
- [ ] Add multi-language support
- [ ] Implement user authentication
- [ ] Add case tracking and notifications
- [ ] Create web UI dashboard