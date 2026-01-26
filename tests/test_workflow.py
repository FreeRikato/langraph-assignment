"""Integration tests for the conservation agent workflow."""

import pytest

from agents.graph import app
from agents.state import ConservationAgentState


class TestWorkflowIntegration:
    """Integration tests for end-to-end workflows."""

    def make_state(self, query: str) -> ConservationAgentState:
        """Helper to create an initial state."""
        return {
            "query": query,
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "language": "English",
            "visualization_data": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

    def test_species_info_workflow(self):
        """Test Case A: Simple species lookup."""
        state = self.make_state("Tell me about the Javan Rhino")

        # Provide config for MemorySaver checkpointer
        config = {"configurable": {"thread_id": "test_thread_1"}}
        result = app.invoke(state, config=config)

        # Verify response was generated
        assert "final_response" in result
        assert len(result["final_response"]) > 0

        # Verify workflow steps were executed
        assert result["intent"] in ["species_info", "general_chat"]
        assert "retrieved_data" in result
        assert "analysis_result" in result

    def test_threat_analysis_workflow(self):
        """Test Case B: Threat analysis query."""
        state = self.make_state("What are the threats to African Elephants?")

        config = {"configurable": {"thread_id": "test_thread_2"}}
        result = app.invoke(state, config=config)

        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "threat" in result["final_response"].lower() or result["intent"] != "threat_analysis"

    def test_funding_analysis_workflow(self):
        """Test Case C: Funding analysis query."""
        state = self.make_state("Which animal groups receive the least funding?")

        config = {"configurable": {"thread_id": "test_thread_3"}}
        result = app.invoke(state, config=config)

        assert "final_response" in result
        assert len(result["final_response"]) > 0

    def test_comparative_analysis_workflow(self):
        """Test Case D: Comparative analysis query."""
        state = self.make_state("Compare funding for Rhinos vs Pandas")

        config = {"configurable": {"thread_id": "test_thread_4"}}
        result = app.invoke(state, config=config)

        assert "final_response" in result
        assert len(result["final_response"]) > 0

    def test_state_structure_remains_valid(self):
        """Test that state structure remains valid throughout workflow."""
        state = self.make_state("Tell me about pandas")

        config = {"configurable": {"thread_id": "test_thread_5"}}
        result = app.invoke(state, config=config)

        # Verify all expected keys are present
        expected_keys = [
            "query", "conversation_history", "intent", "entities",
            "retrieved_data", "current_tool", "tool_results",
            "analysis_result", "final_response", "iteration_count",
            "language", "visualization_data"
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_iteration_counter_increments(self):
        """Test that iteration counter is tracked."""
        state = self.make_state("What about snow leopards?")

        config = {"configurable": {"thread_id": "test_thread_6"}}
        result = app.invoke(state, config=config)

        # Iteration count should be present
        assert "iteration_count" in result
        assert isinstance(result["iteration_count"], int)

    def test_general_chat_handling(self):
        """Test that general chat queries are handled."""
        state = self.make_state("Hello, how are you?")

        config = {"configurable": {"thread_id": "test_thread_7"}}
        result = app.invoke(state, config=config)

        # Should still generate a response
        assert "final_response" in result
        assert len(result["final_response"]) > 0


class TestToolExecution:
    """Tests for tool execution within the workflow."""

    def make_state(self, query: str) -> ConservationAgentState:
        """Helper to create an initial state."""
        return {
            "query": query,
            "conversation_history": [],
            "intent": "",
            "entities": [],
            "retrieved_data": {},
            "current_tool": "",
            "tool_results": {},
            "analysis_result": "",
            "final_response": "",
            "language": "English",
            "visualization_data": "",
            "iteration_count": 0,
            "max_iterations": 3,
        }

    def test_species_tool_returns_data(self):
        """Test that species info tool returns data."""
        state = self.make_state("Tell me about the African Elephant")

        config = {"configurable": {"thread_id": "test_thread_8"}}
        result = app.invoke(state, config=config)

        # Tool results should be populated
        if result["current_tool"] == "retrieve_species_info":
            assert "tool_results" in result
            assert len(result["tool_results"]) > 0

    def test_migration_tool_execution(self):
        """Test that migration tool executes correctly."""
        state = self.make_state("How far do wildebeest migrate?")

        config = {"configurable": {"thread_id": "test_thread_9"}}
        result = app.invoke(state, config=config)

        assert "final_response" in result
        assert len(result["final_response"]) > 0

    def test_error_handling_in_tools(self):
        """Test that tool errors are handled gracefully."""
        # Use a non-existent species
        state = self.make_state("Tell me about the unicorn")

        config = {"configurable": {"thread_id": "test_thread_10"}}
        result = app.invoke(state, config=config)

        # Should still generate a response even if data not found
        assert "final_response" in result
        assert len(result["final_response"]) > 0
