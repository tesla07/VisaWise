"""LangGraph workflow agent for stateful USCIS processing."""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage
import operator
import logging

from ..services import USCISService
from ..config import settings


logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State for the USCIS workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    receipt_number: str
    case_status: dict
    user_query: str
    response: str
    needs_clarification: bool


class USCISWorkflowAgent:
    """LangGraph-based workflow agent for USCIS query processing."""
    
    def __init__(self, uscis_service: USCISService):
        """Initialize the workflow agent.
        
        Args:
            uscis_service: USCIS service instance
        """
        self.uscis_service = uscis_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the stateful workflow graph.
        
        Returns:
            StateGraph instance
        """
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("check_status", self._check_status)
        workflow.add_node("analyze_status", self._analyze_status)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("clarify", self._clarify)
        
        # Add edges
        workflow.set_entry_point("parse_query")
        
        workflow.add_conditional_edges(
            "parse_query",
            self._should_check_status,
            {
                "check": "check_status",
                "clarify": "clarify",
                "direct": "generate_response"
            }
        )
        
        workflow.add_edge("check_status", "analyze_status")
        workflow.add_edge("analyze_status", "generate_response")
        workflow.add_edge("clarify", END)
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def _parse_query(self, state: WorkflowState) -> WorkflowState:
        """Parse the user query to extract intent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        query = state.get("user_query", "")
        
        # Use LLM to extract receipt number if present
        prompt = f"""Extract the USCIS receipt number from this query if present.
Receipt numbers are typically 13 characters: 3 letters + 10 digits (e.g., WAC2190012345).

Query: {query}

If a receipt number is found, respond with just the receipt number.
If not found, respond with "NONE".
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        extracted = response.content.strip()
        
        if extracted != "NONE" and len(extracted) == 13:
            state["receipt_number"] = extracted
            state["needs_clarification"] = False
        else:
            state["needs_clarification"] = "receipt" in query.lower() or "status" in query.lower()
        
        return state
    
    def _should_check_status(self, state: WorkflowState) -> str:
        """Determine if we should check case status.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node to visit
        """
        if state.get("receipt_number"):
            return "check"
        elif state.get("needs_clarification"):
            return "clarify"
        else:
            return "direct"
    
    async def _check_status(self, state: WorkflowState) -> WorkflowState:
        """Check the case status.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        receipt_number = state.get("receipt_number", "")
        if receipt_number:
            status = await self.uscis_service.check_case_status(receipt_number)
            state["case_status"] = status
        return state
    
    async def _analyze_status(self, state: WorkflowState) -> WorkflowState:
        """Analyze the case status and determine response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        case_status = state.get("case_status", {})
        
        prompt = f"""Analyze this USCIS case status and provide a helpful explanation:

Receipt Number: {state.get('receipt_number')}
Status: {case_status.get('status')}
Success: {case_status.get('success')}

Provide a clear, empathetic explanation of what this status means and what the applicant should do next.
If there's an error, explain it clearly.
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["response"] = response.content
        
        return state
    
    async def _generate_response(self, state: WorkflowState) -> WorkflowState:
        """Generate the final response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        if state.get("response"):
            return state
        
        # Generate response for queries without status check
        query = state.get("user_query", "")
        prompt = f"""You are a helpful USCIS immigration assistant. Answer this query:

{query}

Provide accurate, helpful information. Remind the user to check official USCIS resources 
or consult an immigration attorney for legal advice.
"""
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        state["response"] = response.content
        
        return state
    
    async def _clarify(self, state: WorkflowState) -> WorkflowState:
        """Request clarification from the user.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        state["response"] = (
            "I'd be happy to help check your case status! "
            "Please provide your 13-character USCIS receipt number "
            "(e.g., WAC2190012345)."
        )
        return state
    
    async def process(self, user_query: str) -> str:
        """Process a user query through the workflow.
        
        Args:
            user_query: User's question or request
            
        Returns:
            Response
        """
        try:
            initial_state = {
                "messages": [],
                "user_query": user_query,
                "receipt_number": "",
                "case_status": {},
                "response": "",
                "needs_clarification": False
            }
            
            result = await self.workflow.ainvoke(initial_state)
            return result.get("response", "I'm sorry, I couldn't process your request.")
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
