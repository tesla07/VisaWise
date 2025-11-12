# VisaWise Architecture Documentation

## System Overview

VisaWise is a multi-layered AI-powered system designed to provide intelligent assistance for USCIS immigration queries. It integrates multiple state-of-the-art AI frameworks and observability tools to deliver reliable, scalable, and monitored service.

## Core Architecture

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         Client Layer                           │
│  (HTTP/REST API Clients, Web Apps, Mobile Apps)               │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                      API Gateway Layer                         │
│                         (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Routes: /query, /case-status, /health, /metrics     │    │
│  │  Middleware: CORS, Authentication (future)           │    │
│  │  Request Validation: Pydantic Models                 │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                      Agent Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  LangChain   │  │  LangGraph   │  │   CrewAI     │       │
│  │   Agent      │  │  Workflow    │  │   Agents     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                 │
│         └─────────────────┴─────────────────┘                 │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                    Service Layer                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            USCIS Integration Service                    │  │
│  │  - Receipt Number Validation                           │  │
│  │  - Case Status Checking                                │  │
│  │  - Response Parsing                                    │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                   Cross-Cutting Concerns                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │    MCP     │  │ Monitoring │  │   Caching  │             │
│  │   Server   │  │(Prometheus)│  │   (Redis)  │             │
│  └────────────┘  └────────────┘  └────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Gateway Layer (FastAPI)

**Location**: `src/visawise/api/app.py`

**Responsibilities**:
- HTTP request handling
- Request/response validation
- Routing to appropriate agents
- Error handling and logging
- Metrics collection

**Key Endpoints**:
- `POST /query`: Main query processing endpoint
- `POST /case-status`: Direct case status checks
- `GET /health`: Health check
- `GET /metrics`: Prometheus metrics
- `GET /mcp/tools`: MCP tool listing
- `GET /mcp/context/{session_id}`: Session context management

**Technology Stack**:
- FastAPI: Async web framework
- Pydantic: Request/response validation
- Uvicorn: ASGI server

### 2. Agent Layer

#### 2.1 LangChain Agent

**Location**: `src/visawise/agents/langchain_agent.py`

**Architecture**:
```
LangChainUSCISAgent
    ├── OpenAI LLM (GPT-4)
    ├── Tools
    │   ├── check_case_status
    │   ├── explain_visa_process
    │   └── interpret_status
    └── AgentExecutor
```

**Features**:
- Function calling with OpenAI
- Tool-based architecture
- Conversation memory
- Structured prompts

**Use Cases**:
- Direct case status checks
- Visa process explanations
- Status interpretations

#### 2.2 LangGraph Workflow

**Location**: `src/visawise/agents/langgraph_workflow.py`

**Architecture**:
```
StateGraph
    ├── parse_query (entry point)
    ├── check_status
    ├── analyze_status
    ├── generate_response
    └── clarify

State: {
    messages, receipt_number, case_status,
    user_query, response, needs_clarification
}
```

**Features**:
- Stateful workflow execution
- Conditional routing
- Context preservation
- Multi-step processing

**Use Cases**:
- Complex queries requiring clarification
- Multi-step workflows
- Context-dependent interactions

#### 2.3 CrewAI Agents

**Location**: `src/visawise/agents/crewai_agents.py`

**Architecture**:
```
USCISCrewAgents
    ├── Status Specialist
    │   └── Expert in case status interpretation
    ├── Process Expert
    │   └── Expert in immigration processes
    ├── Router
    │   └── Query routing and intent detection
    └── Synthesizer
        └── Response combination and formatting

Tasks executed sequentially:
1. Route query
2. Check status (if needed)
3. Get process info (if needed)
4. Synthesize response
```

**Features**:
- Multi-agent collaboration
- Specialized expertise
- Sequential task execution
- Delegation support

**Use Cases**:
- Comprehensive questions
- Multi-aspect queries
- Complex immigration scenarios

### 3. Service Layer

#### USCIS Service

**Location**: `src/visawise/services/uscis_service.py`

**Architecture**:
```
USCISService
    ├── AsyncClient (httpx)
    ├── Receipt Number Validator
    ├── Status Checker
    └── Response Parser
```

**Methods**:
- `check_case_status(receipt_number)`: Check status
- `_validate_receipt_number(receipt_number)`: Validate format
- `_parse_response(html, receipt_number)`: Parse USCIS HTML

**Validation Rules**:
- Format: 3 letters + 10 digits (e.g., WAC2190012345)
- Regex: `^[A-Z]{3}\d{10}$`

**Status Detection**:
- Case Was Received → "Received"
- Case Was Approved → "Approved"
- Request for Additional Evidence → "RFE Issued"
- Case Was Denied → "Denied"
- Case Was Transferred → "Transferred"

### 4. Cross-Cutting Concerns

#### 4.1 MCP (Model Context Protocol) Server

**Location**: `src/visawise/mcp/server.py`

**Purpose**: Enable AI model context sharing and tool registration

**Features**:
- Tool registration and discovery
- Session context management
- Context export/import
- JSON serialization

**Methods**:
- `get_tools()`: List available tools
- `add_context(session_id, context)`: Store context
- `get_context(session_id)`: Retrieve context
- `update_context(session_id, updates)`: Update context
- `clear_context(session_id)`: Clear context

#### 4.2 Monitoring and Metrics

**Location**: `src/visawise/utils/monitoring.py`

**Metrics Collected**:

1. **Counters**:
   - `visawise_queries_total{agent_type, status}`: Total queries
   - `visawise_case_checks_total{status}`: Case status checks
   - `visawise_errors_total{error_type}`: Errors

2. **Histograms**:
   - `visawise_query_duration_seconds{agent_type}`: Query latency
   - `visawise_case_check_duration_seconds`: Check latency

3. **Gauges**:
   - `visawise_active_sessions`: Active sessions
   - `visawise_cache_size`: Cache entries

**Integration**:
- Prometheus scrapes `/metrics` endpoint
- Grafana visualizes metrics
- Custom dashboards for monitoring

#### 4.3 Configuration Management

**Location**: `src/visawise/config/settings.py`

**Settings**:
```python
Settings (Pydantic BaseSettings)
    ├── OpenAI Configuration
    │   ├── openai_api_key
    │   └── openai_model
    ├── USCIS Configuration
    │   └── uscis_api_base_url
    ├── Redis Configuration
    │   ├── redis_host
    │   ├── redis_port
    │   └── redis_db
    ├── API Configuration
    │   ├── api_host
    │   └── api_port
    └── Monitoring Configuration
        ├── grafana_host/port
        └── prometheus_port
```

**Features**:
- Environment variable loading
- Type validation
- Default values
- Centralized configuration

## Data Flow

### Query Processing Flow

```
1. Client Request
   └─> POST /query {query, agent_type, session_id}

2. API Gateway
   ├─> Validate request (Pydantic)
   ├─> Select agent (langchain/langgraph/crewai)
   └─> Start timer

3. Agent Processing
   ├─> Parse query
   ├─> Determine intent
   ├─> Check case status (if needed)
   │   └─> USCIS Service
   │       ├─> Validate receipt number
   │       ├─> HTTP request to USCIS
   │       └─> Parse response
   ├─> Generate response
   └─> Return result

4. Post-Processing
   ├─> Update MCP context (if session_id)
   ├─> Record metrics
   └─> Return response to client

5. Client Response
   └─> {response, agent_type, processing_time, session_id}
```

### Case Status Check Flow

```
1. Client Request
   └─> POST /case-status {receipt_number}

2. USCIS Service
   ├─> Validate receipt number format
   ├─> Make HTTP POST to USCIS API
   ├─> Parse HTML response
   └─> Extract status information

3. Response
   └─> {success, receipt_number, status, description, timestamp}
```

## Deployment Architecture

### Docker Compose Setup

```yaml
Services:
  ├── visawise-api
  │   ├── Port: 8000
  │   └── Depends: redis, prometheus
  ├── redis
  │   └── Port: 6379
  ├── prometheus
  │   └── Port: 9090
  └── grafana
      └── Port: 3000
```

### Scaling Considerations

1. **Horizontal Scaling**:
   - API service can be scaled horizontally
   - Use load balancer for distribution
   - Shared Redis for session state

2. **Vertical Scaling**:
   - Increase container resources
   - Adjust worker count (Uvicorn)

3. **Caching**:
   - Redis for case status caching
   - Reduce USCIS API calls
   - TTL-based invalidation

## Security Architecture

### Current Implementation

1. **Input Validation**:
   - Pydantic models for request validation
   - Receipt number format validation
   - Type checking

2. **Dependency Security**:
   - Regular vulnerability scanning
   - Updated to patched versions
   - No known vulnerabilities (CodeQL verified)

3. **API Security**:
   - CORS configured
   - Request size limits
   - Rate limiting (future)

### Future Enhancements

1. **Authentication**: JWT tokens, OAuth2
2. **Authorization**: Role-based access control
3. **Encryption**: TLS/SSL for API, encrypted secrets
4. **Rate Limiting**: Per-user, per-IP limits
5. **Audit Logging**: Security event tracking

## Performance Considerations

### Current Performance

1. **Response Times**:
   - Simple queries: <2s
   - Case status checks: <1s
   - Complex workflows: 2-5s

2. **Concurrency**:
   - AsyncIO for non-blocking I/O
   - Concurrent request handling
   - Connection pooling (httpx)

3. **Caching Strategy**:
   - Redis for session state
   - Future: Case status caching

### Optimization Strategies

1. **Agent Selection**:
   - Use simplest agent for task
   - LangChain for simple queries
   - LangGraph for complex workflows
   - CrewAI for comprehensive analysis

2. **Resource Management**:
   - Connection pooling
   - Request timeouts
   - Circuit breakers (future)

3. **Monitoring**:
   - Track slow queries
   - Identify bottlenecks
   - Optimize hot paths

## Testing Architecture

### Test Structure

```
tests/
├── test_uscis_service.py (Unit tests)
│   ├── Validation tests
│   ├── Parsing tests
│   └── Error handling tests
├── test_mcp.py (Integration tests)
│   ├── Context management
│   └── Tool registration
├── test_monitoring.py (Unit tests)
│   └── Metrics recording
└── test_api.py (Integration tests)
    ├── Endpoint tests
    └── E2E workflows
```

### Test Coverage

- Unit tests: Core logic
- Integration tests: Component interaction
- E2E tests: Full workflows (future)

## Monitoring and Observability

### Metrics Dashboard

Grafana panels monitor:
1. Request rate by agent type
2. Response time percentiles (p50, p95)
3. Error rates by type
4. Active sessions
5. Cache utilization
6. USCIS API latency

### Alerting (Future)

1. High error rate (>5%)
2. Slow response times (p95 >5s)
3. Service unavailability
4. API quota exhaustion

## Future Architecture Evolution

### Phase 2 Enhancements

1. **Advanced Caching**:
   - Redis-based case status cache
   - Intelligent cache invalidation
   - Multi-level caching

2. **Real-time Updates**:
   - WebSocket support
   - Push notifications
   - Case status subscriptions

3. **Enhanced Analytics**:
   - User behavior tracking
   - Query pattern analysis
   - Predictive insights

4. **Multi-language Support**:
   - Translation layer
   - Localized responses
   - Cultural adaptations

### Phase 3 Enhancements

1. **Advanced AI**:
   - Fine-tuned models
   - RAG (Retrieval Augmented Generation)
   - Knowledge base integration

2. **Web UI**:
   - React/Vue frontend
   - Interactive dashboards
   - Case tracking interface

3. **Mobile App**:
   - iOS/Android apps
   - Push notifications
   - Offline support

## References

- FastAPI: https://fastapi.tiangolo.com/
- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- CrewAI: https://www.crewai.com/
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/
