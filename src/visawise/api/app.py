"""FastAPI application for VisaWise."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
import logging
from contextlib import asynccontextmanager

from ..config import settings
from ..services import USCISService
from ..agents import LangChainUSCISAgent, USCISWorkflowAgent, USCISCrewAgents
from ..mcp import mcp_server
from ..utils import setup_metrics, get_metrics


logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User's query or question")
    agent_type: str = Field(
        default="langgraph",
        description="Agent type to use: langchain, langgraph, or crewai"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context tracking"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    response: str = Field(..., description="Agent's response")
    agent_type: str = Field(..., description="Agent type used")
    processing_time: float = Field(..., description="Processing time in seconds")
    session_id: Optional[str] = None


class CaseStatusRequest(BaseModel):
    """Request model for case status check."""
    receipt_number: str = Field(..., description="USCIS receipt number")


class CaseStatusResponse(BaseModel):
    """Response model for case status check."""
    success: bool
    receipt_number: str
    status: Optional[str] = None
    description: Optional[str] = None
    case_type: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    services: Dict[str, str]


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting VisaWise API...")
    
    # Initialize services
    app.state.uscis_service = USCISService(settings.uscis_api_base_url)
    app.state.langchain_agent = LangChainUSCISAgent(app.state.uscis_service)
    app.state.langgraph_agent = USCISWorkflowAgent(app.state.uscis_service)
    app.state.crewai_agents = USCISCrewAgents(app.state.uscis_service)
    
    # Initialize metrics
    app.state.metrics = setup_metrics()
    
    logger.info("VisaWise API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VisaWise API...")
    await app.state.uscis_service.close()
    logger.info("VisaWise API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="VisaWise API",
    description="USCIS Query Resolution System with AI Agents",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service info."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "langchain": "active",
            "langgraph": "active",
            "crewai": "active",
            "mcp": "active",
            "prometheus": "active"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "api": "up",
            "uscis_service": "up",
            "agents": "up"
        }
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, req: Request):
    """Process a user query using the specified agent.
    
    Args:
        request: Query request
        req: FastAPI request object
        
    Returns:
        Query response
    """
    start_time = time.time()
    metrics = get_metrics()
    
    try:
        # Get the appropriate agent
        if request.agent_type == "langchain":
            agent = req.app.state.langchain_agent
            response = await agent.query(request.query)
        elif request.agent_type == "langgraph":
            agent = req.app.state.langgraph_agent
            response = await agent.process(request.query)
        elif request.agent_type == "crewai":
            agent = req.app.state.crewai_agents
            response = await agent.process_query(request.query)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_type: {request.agent_type}"
            )
        
        # Track context in MCP if session_id provided
        if request.session_id:
            mcp_server.update_context(request.session_id, {
                "last_query": request.query,
                "last_response": response,
                "agent_type": request.agent_type
            })
        
        processing_time = time.time() - start_time
        
        # Record metrics
        metrics.record_query(request.agent_type, "success", processing_time)
        
        return QueryResponse(
            response=response,
            agent_type=request.agent_type,
            processing_time=processing_time,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        metrics.record_error("query_processing")
        processing_time = time.time() - start_time
        metrics.record_query(request.agent_type, "error", processing_time)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/case-status", response_model=CaseStatusResponse)
async def check_case_status(request: CaseStatusRequest, req: Request):
    """Check USCIS case status directly.
    
    Args:
        request: Case status request
        req: FastAPI request object
        
    Returns:
        Case status response
    """
    start_time = time.time()
    metrics = get_metrics()
    
    try:
        uscis_service = req.app.state.uscis_service
        result = await uscis_service.check_case_status(request.receipt_number)
        
        processing_time = time.time() - start_time
        status = "success" if result.get("success") else "error"
        metrics.record_case_check(status, processing_time)
        
        return CaseStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Error checking case status: {str(e)}")
        metrics.record_error("case_status_check")
        processing_time = time.time() - start_time
        metrics.record_case_check("error", processing_time)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics
    """
    metrics = get_metrics()
    return Response(content=metrics.get_metrics(), media_type="text/plain")


@app.get("/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools.
    
    Returns:
        List of tools
    """
    return {"tools": mcp_server.get_tools()}


@app.get("/mcp/context/{session_id}")
async def get_mcp_context(session_id: str):
    """Get MCP context for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Context data
    """
    context = mcp_server.get_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "context": context}


@app.delete("/mcp/context/{session_id}")
async def clear_mcp_context(session_id: str):
    """Clear MCP context for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    mcp_server.clear_context(session_id)
    return {"message": f"Context cleared for session {session_id}"}
