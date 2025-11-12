"""Tests for MCP server."""

import pytest
from src.visawise.mcp.server import MCPServer


@pytest.fixture
def mcp_server():
    """Create an MCP server instance."""
    return MCPServer()


def test_get_tools(mcp_server):
    """Test getting available tools."""
    tools = mcp_server.get_tools()
    assert len(tools) > 0
    assert all("name" in tool for tool in tools)
    assert all("description" in tool for tool in tools)


def test_add_context(mcp_server):
    """Test adding context."""
    session_id = "test-session-1"
    context = {"user": "test_user", "query": "test query"}
    mcp_server.add_context(session_id, context)
    
    retrieved = mcp_server.get_context(session_id)
    assert retrieved == context


def test_get_context_not_found(mcp_server):
    """Test getting context that doesn't exist."""
    context = mcp_server.get_context("non-existent")
    assert context == {}


def test_update_context(mcp_server):
    """Test updating context."""
    session_id = "test-session-2"
    initial = {"user": "test_user"}
    mcp_server.add_context(session_id, initial)
    
    updates = {"query": "updated query"}
    mcp_server.update_context(session_id, updates)
    
    retrieved = mcp_server.get_context(session_id)
    assert retrieved["user"] == "test_user"
    assert retrieved["query"] == "updated query"


def test_clear_context(mcp_server):
    """Test clearing context."""
    session_id = "test-session-3"
    mcp_server.add_context(session_id, {"test": "data"})
    mcp_server.clear_context(session_id)
    
    context = mcp_server.get_context(session_id)
    assert context == {}


def test_export_import_context(mcp_server):
    """Test exporting and importing context."""
    session_id = "test-session-4"
    original_context = {"user": "test_user", "data": [1, 2, 3]}
    mcp_server.add_context(session_id, original_context)
    
    # Export
    exported = mcp_server.export_context(session_id)
    assert isinstance(exported, str)
    
    # Clear and import
    mcp_server.clear_context(session_id)
    mcp_server.import_context(session_id, exported)
    
    # Verify
    imported = mcp_server.get_context(session_id)
    assert imported == original_context
