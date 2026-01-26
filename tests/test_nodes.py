"""Unit tests for conservation agent nodes."""

import pytest

from agents.nodes import (
    query_processing_node,
    data_retrieval_node,
    tool_selection_node,
)
from agents.state import ConservationAgentState


class TestQueryProcessingNode:
    """Tests for the query processing node."""

    def test_species_intent_extraction(self):
        """Test that species queries are correctly classified."""
        state: ConservationAgentState = {
            "query": "Tell me about the Javan Rhino",
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = query_processing_node(state)

        assert "intent" in result
        assert result["intent"] in ["species_info", "general_chat"]
        assert isinstance(result.get("entities"), list)

    def test_threat_intent_extraction(self):
        """Test that threat queries are correctly classified."""
        state: ConservationAgentState = {
            "query": "What are the main threats to African Elephants?",
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = query_processing_node(state)

        assert "intent" in result
        assert result["intent"] in ["threat_analysis", "general_chat"]

    def test_funding_intent_extraction(self):
        """Test that funding queries are correctly classified."""
        state: ConservationAgentState = {
            "query": "Which animal groups receive the least funding?",
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = query_processing_node(state)

        assert "intent" in result
        assert result["intent"] in ["funding_analysis", "general_chat"]


class TestDataRetrievalNode:
    """Tests for the data retrieval node."""

    def test_data_retrieval_loads_report(self):
        """Test that the conservation report is loaded."""
        state: ConservationAgentState = {
            "query": "Test query",
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = data_retrieval_node(state)

        assert "retrieved_data" in result
        assert "report_context" in result["retrieved_data"]
        # Verify report content exists
        assert len(result["retrieved_data"]["report_context"]) > 0


class TestToolSelectionNode:
    """Tests for the tool selection node."""

    def test_species_info_tool_selection(self):
        """Test that species_info intent maps to retrieve_species_info tool."""
        state: ConservationAgentState = {
            "query": "Tell me about rhinos",
            "conversation_history": [],
            "intent": "species_info",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = tool_selection_node(state)

        assert result["current_tool"] == "retrieve_species_info"

    def test_threat_analysis_tool_selection(self):
        """Test that threat_analysis intent maps to analyze_threats tool."""
        state: ConservationAgentState = {
            "query": "What are the threats?",
            "conversation_history": [],
            "intent": "threat_analysis",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = tool_selection_node(state)

        assert result["current_tool"] == "analyze_threats"

    def test_funding_analysis_tool_selection(self):
        """Test that funding_analysis intent maps to analyze_funding tool."""
        state: ConservationAgentState = {
            "query": "How much funding?",
            "conversation_history": [],
            "intent": "funding_analysis",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = tool_selection_node(state)

        assert result["current_tool"] == "analyze_funding"

    def test_migration_tool_selection(self):
        """Test that migration_lookup intent maps to get_migration_data tool."""
        state: ConservationAgentState = {
            "query": "Where do they migrate?",
            "conversation_history": [],
            "intent": "migration_lookup",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = tool_selection_node(state)

        assert result["current_tool"] == "get_migration_data"

    def test_general_chat_no_tool(self):
        """Test that general_chat intent maps to no tool."""
        state: ConservationAgentState = {
            "query": "Hello",
            "conversation_history": [],
            "intent": "general_chat",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = tool_selection_node(state)

        assert result["current_tool"] == "none"
