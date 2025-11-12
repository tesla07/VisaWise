#!/usr/bin/env python3
"""Example script demonstrating VisaWise API usage."""

import httpx
import asyncio
import json


async def main():
    """Run example queries against VisaWise API."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health check
        print("=" * 60)
        print("1. Health Check")
        print("=" * 60)
        response = await client.get(f"{base_url}/health")
        print(json.dumps(response.json(), indent=2))
        print()
        
        # 2. Query with LangGraph agent
        print("=" * 60)
        print("2. Query with LangGraph Agent")
        print("=" * 60)
        query_data = {
            "query": "What does it mean if my case status is 'RFE Issued'?",
            "agent_type": "langgraph",
            "session_id": "example-session-1"
        }
        response = await client.post(f"{base_url}/query", json=query_data)
        result = response.json()
        print(f"Query: {query_data['query']}")
        print(f"Response: {result['response']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print()
        
        # 3. Check case status (with example receipt number)
        print("=" * 60)
        print("3. Direct Case Status Check")
        print("=" * 60)
        status_data = {
            "receipt_number": "WAC2190012345"
        }
        response = await client.post(f"{base_url}/case-status", json=status_data)
        result = response.json()
        print(json.dumps(result, indent=2))
        print()
        
        # 4. Query with CrewAI agents
        print("=" * 60)
        print("4. Query with CrewAI Agents")
        print("=" * 60)
        query_data = {
            "query": "Can you explain the H-1B visa process and timeline?",
            "agent_type": "crewai",
            "session_id": "example-session-2"
        }
        response = await client.post(f"{base_url}/query", json=query_data)
        result = response.json()
        print(f"Query: {query_data['query']}")
        print(f"Response: {result['response']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print()
        
        # 5. Get MCP tools
        print("=" * 60)
        print("5. Available MCP Tools")
        print("=" * 60)
        response = await client.get(f"{base_url}/mcp/tools")
        print(json.dumps(response.json(), indent=2))
        print()
        
        # 6. Get MCP context
        print("=" * 60)
        print("6. MCP Session Context")
        print("=" * 60)
        response = await client.get(f"{base_url}/mcp/context/example-session-1")
        print(json.dumps(response.json(), indent=2))
        print()
        
        # 7. Check metrics
        print("=" * 60)
        print("7. Prometheus Metrics Sample")
        print("=" * 60)
        response = await client.get(f"{base_url}/metrics")
        lines = response.text.split('\n')[:20]  # Show first 20 lines
        print('\n'.join(lines))
        print("... (metrics output truncated)")
        print()


if __name__ == "__main__":
    print("VisaWise API Example Usage")
    print("Make sure the API is running: python main.py")
    print()
    asyncio.run(main())
