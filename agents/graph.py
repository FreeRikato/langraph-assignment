"""LangGraph workflow definition for the conservation agent.

Defines the StateGraph with all nodes and edges that orchestrate
the agentic workflow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import ConservationAgentState
from agents.nodes import (
    query_processing_node,
    data_retrieval_node,
    tool_selection_node,
    tool_execution_node,
    analysis_node,
    response_generation_node,
)


# Conditional edge functions

def route_to_tool(state: ConservationAgentState) -> str:
    """
    Route from tool selection to either tool execution or direct analysis.

    If a tool is selected, proceed to tool execution.
    If no tool is needed (general chat), skip to analysis.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    current_tool = state.get("current_tool", "none")
    valid_tools = [
        "retrieve_species_info",
        "analyze_threats",
        "analyze_funding",
        "get_migration_data",
        "compare_conservation_metrics",
        "predict_recovery_success",
        "web_search",  # Challenge 1
        "predict_population",  # Challenge 3
    ]

    if current_tool == "none" or current_tool not in valid_tools:
        return "analysis"
    return "tool_execution"


def check_iteration_limit(state: ConservationAgentState) -> str:
    """
    Check if we've exceeded the iteration limit.

    Prevents infinite loops in the agent workflow.

    Args:
        state: Current agent state.

    Returns:
        Next node name (either loop back to tool selection or continue to response).
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        return "response_generation"

    # Check if analysis indicates we need more data
    analysis = state.get("analysis_result", "")
    if "more information" in analysis.lower() or "additional data" in analysis.lower():
        return "tool_selection"

    return "response_generation"


def create_conservation_agent_graph():
    """
    Create and compile the LangGraph workflow for the conservation agent.

    The graph follows this flow:
    1. Query Processing - Parse user query
    2. Data Retrieval - Load conservation report
    3. Tool Selection - Choose appropriate tool
    4. Tool Execution (conditional) - Execute selected tool
    5. Analysis - Generate insights
    6. Response Generation - Format final answer

    Returns:
        Compiled LangGraph application.
    """
    # Create the state graph
    workflow = StateGraph(ConservationAgentState)

    # Add all nodes
    workflow.add_node("query_processing", query_processing_node)
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("tool_selection", tool_selection_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("response_generation", response_generation_node)

    # Define the flow with edges

    # Entry point
    workflow.set_entry_point("query_processing")

    # Standard linear edges
    workflow.add_edge("query_processing", "data_retrieval")
    workflow.add_edge("data_retrieval", "tool_selection")

    # Conditional edge: tool_selection -> tool_execution OR analysis
    workflow.add_conditional_edges(
        "tool_selection",
        route_to_tool,
        {
            "tool_execution": "tool_execution",
            "analysis": "analysis"
        }
    )

    # Standard edge after tool execution
    workflow.add_edge("tool_execution", "analysis")

    # Conditional edge: analysis -> response_generation (or loop back)
    workflow.add_conditional_edges(
        "analysis",
        check_iteration_limit,
        {
            "tool_selection": "tool_selection",
            "response_generation": "response_generation"
        }
    )

    # Final edge to END
    workflow.add_edge("response_generation", END)

    # Compile the graph with MemorySaver for persistent learning (Challenge 2)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# Create the compiled application instance
app = create_conservation_agent_graph()
