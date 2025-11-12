"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient
from src.visawise.api.app import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
async def test_query_endpoint_langgraph():
    """Test query endpoint with langgraph agent."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "query": "What is an H-1B visa?",
            "agent_type": "langgraph"
        }
        response = await client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["agent_type"] == "langgraph"
        assert "processing_time" in data


@pytest.mark.asyncio
async def test_query_endpoint_invalid_agent():
    """Test query endpoint with invalid agent type."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "query": "Test query",
            "agent_type": "invalid_agent"
        }
        response = await client.post("/query", json=payload)
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_case_status_endpoint():
    """Test case status endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "receipt_number": "WAC2190012345"
        }
        response = await client.post("/case-status", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "receipt_number" in data
        assert data["receipt_number"] == "WAC2190012345"


@pytest.mark.asyncio
async def test_mcp_tools_endpoint():
    """Test MCP tools endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test metrics endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
