"""CrewAI collaborative agents for USCIS query resolution."""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

from ..services import USCISService
from ..config import settings


logger = logging.getLogger(__name__)


class USCISCrewAgents:
    """CrewAI-based collaborative agents for USCIS queries."""
    
    def __init__(self, uscis_service: USCISService):
        """Initialize the crew of agents.
        
        Args:
            uscis_service: USCIS service instance
        """
        self.uscis_service = uscis_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
        self.agents = self._create_agents()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create the crew of specialized agents.
        
        Returns:
            Dictionary of agents
        """
        # Case Status Specialist
        status_agent = Agent(
            role="USCIS Case Status Specialist",
            goal="Check and interpret USCIS case statuses accurately",
            backstory=(
                "You are an expert in USCIS case processing with years of experience "
                "interpreting case statuses and explaining them to applicants."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
        
        # Immigration Process Expert
        process_agent = Agent(
            role="Immigration Process Expert",
            goal="Explain immigration processes, requirements, and timelines",
            backstory=(
                "You are an immigration expert who helps people understand visa types, "
                "application processes, required documents, and typical timelines."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
        
        # Query Router
        router_agent = Agent(
            role="Query Router",
            goal="Analyze user queries and route them to appropriate specialists",
            backstory=(
                "You are skilled at understanding user intent and directing queries "
                "to the right expert for the most helpful response."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
        
        # Response Synthesizer
        synthesizer_agent = Agent(
            role="Response Synthesizer",
            goal="Combine information from specialists into clear, helpful responses",
            backstory=(
                "You excel at taking complex information from multiple sources and "
                "creating clear, empathetic responses that help users understand their situation."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        return {
            "status": status_agent,
            "process": process_agent,
            "router": router_agent,
            "synthesizer": synthesizer_agent
        }
    
    async def process_query(self, user_query: str) -> str:
        """Process a user query using the crew of agents.
        
        Args:
            user_query: User's question or request
            
        Returns:
            Response from the crew
        """
        try:
            # Create tasks
            tasks = self._create_tasks(user_query)
            
            # Create and run the crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            logger.error(f"Error in crew processing: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
    
    def _create_tasks(self, user_query: str) -> List[Task]:
        """Create tasks for the crew.
        
        Args:
            user_query: User's question
            
        Returns:
            List of tasks
        """
        # Task 1: Route the query
        route_task = Task(
            description=(
                f"Analyze this user query and determine what type of help they need:\n"
                f"Query: {user_query}\n\n"
                f"Determine if they need:\n"
                f"1. Case status check (if they provide or ask about a receipt number)\n"
                f"2. Immigration process information (if they ask about visa types, processes, requirements)\n"
                f"3. Both\n"
                f"4. General guidance"
            ),
            agent=self.agents["router"],
            expected_output="Analysis of query type and needed information"
        )
        
        # Task 2: Get case status if needed
        status_task = Task(
            description=(
                f"If the user query involves a case status check:\n"
                f"Query: {user_query}\n\n"
                f"Extract the receipt number and explain what the status means.\n"
                f"If no receipt number is found, ask the user to provide it."
            ),
            agent=self.agents["status"],
            expected_output="Case status information and interpretation"
        )
        
        # Task 3: Provide process information if needed
        process_task = Task(
            description=(
                f"If the user query involves immigration processes:\n"
                f"Query: {user_query}\n\n"
                f"Provide relevant information about visa types, application processes, "
                f"requirements, and timelines."
            ),
            agent=self.agents["process"],
            expected_output="Immigration process information"
        )
        
        # Task 4: Synthesize response
        synthesize_task = Task(
            description=(
                f"Combine all the information gathered about this query:\n"
                f"Query: {user_query}\n\n"
                f"Create a clear, helpful, and empathetic response that addresses the user's needs.\n"
                f"Remind them to check official USCIS resources or consult an immigration attorney."
            ),
            agent=self.agents["synthesizer"],
            expected_output="Final comprehensive response to the user"
        )
        
        return [route_task, status_task, process_task, synthesize_task]
