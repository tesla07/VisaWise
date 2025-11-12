"""LangChain agent for USCIS query resolution."""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List
import logging

from ..services import USCISService
from ..config import settings


logger = logging.getLogger(__name__)


class LangChainUSCISAgent:
    """LangChain-based agent for USCIS query resolution."""
    
    def __init__(self, uscis_service: USCISService):
        """Initialize the LangChain agent.
        
        Args:
            uscis_service: USCIS service instance
        """
        self.uscis_service = uscis_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent.
        
        Returns:
            List of tools
        """
        async def check_case_status_wrapper(receipt_number: str) -> str:
            """Wrapper for case status checking."""
            result = await self.uscis_service.check_case_status(receipt_number)
            if result.get("success"):
                return f"Case {receipt_number} status: {result.get('status')}. {result.get('description', '')}"
            else:
                return f"Error checking case {receipt_number}: {result.get('error')}"
        
        return [
            Tool(
                name="check_case_status",
                func=lambda x: "Use async version",
                coroutine=check_case_status_wrapper,
                description="Check the status of a USCIS case using the receipt number. "
                           "Input should be a valid receipt number (e.g., WAC2190012345)."
            ),
            Tool(
                name="explain_visa_process",
                func=self._explain_visa_process,
                description="Explain visa processes, requirements, and timelines. "
                           "Input should be the visa type or process question."
            ),
            Tool(
                name="interpret_status",
                func=self._interpret_status,
                description="Interpret what a specific case status means and provide next steps. "
                           "Input should be the status name (e.g., 'RFE Issued', 'Approved')."
            )
        ]
    
    def _create_agent(self):
        """Create the OpenAI functions agent.
        
        Returns:
            Agent instance
        """
        system_message = """You are a helpful USCIS immigration assistant. You help users:
1. Check their case status using receipt numbers
2. Understand visa processes and requirements
3. Interpret case statuses and provide guidance on next steps

Be professional, accurate, and empathetic. Always remind users that you provide information 
but they should consult official USCIS resources or immigration attorneys for legal advice.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(self.llm, self.tools, prompt)
    
    def _explain_visa_process(self, query: str) -> str:
        """Explain visa processes.
        
        Args:
            query: Question about visa process
            
        Returns:
            Explanation
        """
        # This would typically call the LLM with specific context
        return f"Providing information about: {query}. Please check USCIS official website for the most current information."
    
    def _interpret_status(self, status: str) -> str:
        """Interpret a case status.
        
        Args:
            status: Case status
            
        Returns:
            Interpretation
        """
        interpretations = {
            "Received": "Your case has been received by USCIS and is in the initial processing stage.",
            "RFE Issued": "USCIS has issued a Request for Evidence. You need to submit additional documentation.",
            "Approved": "Congratulations! Your case has been approved.",
            "Denied": "Your case has been denied. You may be able to appeal or file a motion to reopen/reconsider.",
            "Transferred": "Your case has been transferred to another USCIS office for processing."
        }
        return interpretations.get(status, f"Status '{status}' requires review of your case details.")
    
    async def query(self, user_input: str) -> str:
        """Process a user query.
        
        Args:
            user_input: User's question or request
            
        Returns:
            Agent's response
        """
        try:
            result = await self.agent_executor.ainvoke({"input": user_input})
            return result.get("output", "I'm sorry, I couldn't process your request.")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
