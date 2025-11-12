"""MCP (Model Context Protocol) server implementation."""

from typing import Dict, Any, List
import logging
import json


logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server for VisaWise to enable AI model context sharing."""
    
    def __init__(self):
        """Initialize MCP server."""
        self.contexts: Dict[str, Any] = {}
        self.tools: List[Dict[str, Any]] = []
        self._register_tools()
    
    def _register_tools(self):
        """Register available tools with MCP."""
        self.tools = [
            {
                "name": "check_case_status",
                "description": "Check the status of a USCIS case using receipt number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receipt_number": {
                            "type": "string",
                            "description": "USCIS receipt number (e.g., WAC2190012345)"
                        }
                    },
                    "required": ["receipt_number"]
                }
            },
            {
                "name": "explain_visa_process",
                "description": "Explain visa processes, requirements, and timelines",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "visa_type": {
                            "type": "string",
                            "description": "Type of visa or process to explain"
                        }
                    },
                    "required": ["visa_type"]
                }
            },
            {
                "name": "get_processing_times",
                "description": "Get current USCIS processing times for different case types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "form_type": {
                            "type": "string",
                            "description": "USCIS form type (e.g., I-129, I-140, I-485)"
                        },
                        "service_center": {
                            "type": "string",
                            "description": "USCIS service center"
                        }
                    },
                    "required": ["form_type"]
                }
            }
        ]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools.
        
        Returns:
            List of tool definitions
        """
        return self.tools
    
    def add_context(self, session_id: str, context: Dict[str, Any]):
        """Add context for a session.
        
        Args:
            session_id: Unique session identifier
            context: Context data to store
        """
        self.contexts[session_id] = context
        logger.info(f"Added context for session {session_id}")
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context data or empty dict if not found
        """
        return self.contexts.get(session_id, {})
    
    def update_context(self, session_id: str, updates: Dict[str, Any]):
        """Update context for a session.
        
        Args:
            session_id: Session identifier
            updates: Context updates to apply
        """
        if session_id in self.contexts:
            self.contexts[session_id].update(updates)
        else:
            self.contexts[session_id] = updates
        logger.info(f"Updated context for session {session_id}")
    
    def clear_context(self, session_id: str):
        """Clear context for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info(f"Cleared context for session {session_id}")
    
    def export_context(self, session_id: str) -> str:
        """Export context as JSON string.
        
        Args:
            session_id: Session identifier
            
        Returns:
            JSON string of context
        """
        context = self.get_context(session_id)
        return json.dumps(context, indent=2)
    
    def import_context(self, session_id: str, context_json: str):
        """Import context from JSON string.
        
        Args:
            session_id: Session identifier
            context_json: JSON string of context
        """
        context = json.loads(context_json)
        self.add_context(session_id, context)


# Global MCP server instance
mcp_server = MCPServer()
