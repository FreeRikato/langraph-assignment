"""Agent state definition for the LangGraph conservation agent."""

from typing import TypedDict, List, Dict, Any, Annotated
import operator


class ConservationAgentState(TypedDict):
    """
    State for the conservation agent workflow.

    This state is passed between nodes in the LangGraph and tracks
    all information needed for query processing, tool execution, and
    response generation.
    """

    # Input fields
    query: str  # User's question about animal conservation

    # Conversation tracking
    conversation_history: Annotated[List[Dict[str, Any]], operator.add]

    # Query analysis fields (populated by query_processing_node)
    intent: str  # One of: species_info, threat_analysis, funding_analysis,
                 # migration_lookup, comparative_analysis, recovery_prediction, general_chat
    entities: List[str]  # Extracted species names, regions, etc.

    # Data fields (populated by data_retrieval_node)
    retrieved_data: Dict[str, Any]  # Content from conservation report

    # Tool execution fields (populated by tool_selection_node and tool_execution_node)
    current_tool: str  # Name of the selected tool
    tool_results: Dict[str, Any]  # Results from tool execution

    # Analysis fields (populated by analysis_node)
    analysis_result: str  # LLM's analysis of the data

    # Output field (populated by response_generation_node)
    final_response: str  # Formatted response to the user

    # Bonus challenge fields
    language: str  # Target language for response (default: "English") - Challenge 8
    visualization_data: str  # Base64 encoded chart (matplotlib) - Challenge 4

    # Flow control
    iteration_count: int  # Track iterations to prevent infinite loops
    max_iterations: int  # Maximum allowed iterations (default: 3)
