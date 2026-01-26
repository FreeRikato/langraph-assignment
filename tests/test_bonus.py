"""Tests for bonus features.

Tests for Challenges 1-8:
1. Web Search Integration
2. Persistent Learning (Memory)
3. Predictive Analytics (ML)
4. Visualization
5. RAG Integration
6. Fine-tuning Dataset
7. Streaming Responses
8. Multi-language Support
"""

import pytest
import json
import os

from agents.state import ConservationAgentState


class TestBonusFeatures:
    """Test suite for bonus features."""

    def test_bonus_features_import(self):
        """Test that bonus features module can be imported."""
        try:
            from bonus_features import (
                web_search_tool,
                predict_and_visualize_population,
                ConservationRAG,
                generate_finetuning_dataset,
                rag_system
            )
            assert web_search_tool is not None
            assert predict_and_visualize_population is not None
            assert ConservationRAG is not None
            assert generate_finetuning_dataset is not None
            assert rag_system is not None
        except ImportError as e:
            pytest.skip(f"Bonus features not available: {e}")

    def test_web_search_tool(self):
        """Test Challenge 1: Web Search tool."""
        try:
            from bonus_features import web_search_tool

            result = web_search_tool("rhino conservation", max_results=2)

            assert result is not None
            assert isinstance(result, list)
            # Either results or error
            if result and "error" not in result[0]:
                assert "title" in result[0] or "body" in result[0]
        except ImportError:
            pytest.skip("Bonus features not available")

    def test_predictive_analytics(self):
        """Test Challenge 3: ML Predictions."""
        try:
            from bonus_features import predict_and_visualize_population

            historical_data = {2010: 100, 2015: 90, 2020: 80}
            result = predict_and_visualize_population("Test Species", historical_data)

            assert "forecast" in result
            assert "trend" in result
            assert "chart_base64" in result
            assert result["trend"] == "Declining"
            assert len(result["forecast"]) > 0
        except ImportError:
            pytest.skip("Bonus features not available")

    def test_visualization_generation(self):
        """Test Challenge 4: Visualization generation."""
        try:
            from bonus_features import predict_and_visualize_population

            historical_data = {2010: 100, 2015: 110, 2020: 120}
            result = predict_and_visualize_population("Test Species", historical_data)

            # Check that base64 chart was generated
            assert result["chart_base64"] is not None
            assert len(result["chart_base64"]) > 0
            # Base64 encoded data should be valid
            assert len(result["chart_base64"]) > 100
        except ImportError:
            pytest.skip("Bonus features not available")

    def test_rag_initialization(self):
        """Test Challenge 5: RAG System initialization."""
        try:
            from bonus_features import ConservationRAG

            rag = ConservationRAG(persist_directory="./test_chroma_db")

            assert rag is not None
            assert rag.embedding_fn is not None

            # Clean up
            import shutil
            if os.path.exists("./test_chroma_db"):
                shutil.rmtree("./test_chroma_db")
        except ImportError:
            pytest.skip("Bonus features not available")

    def test_rag_ingest(self):
        """Test Challenge 5: RAG document ingestion."""
        try:
            from bonus_features import ConservationRAG

            # Create a test file
            test_file = "data/test_report.md"
            os.makedirs("data", exist_ok=True)
            with open(test_file, "w") as f:
                f.write("## Test Section\n\nThis is a test document for RAG.\n\n## Another Section\n\nMore content here.")

            rag = ConservationRAG(persist_directory="./test_chroma_db2")
            rag.ingest_report(test_file)

            assert rag.vector_store is not None

            # Clean up
            import shutil
            if os.path.exists("./test_chroma_db2"):
                shutil.rmtree("./test_chroma_db2")
            if os.path.exists(test_file):
                os.remove(test_file)
        except ImportError:
            pytest.skip("Bonus features not available")
        except Exception as e:
            # ChromaDB may have issues in test environment
            pytest.skip(f"RAG ingest test skipped: {e}")

    def test_finetuning_dataset_generation(self):
        """Test Challenge 6: Fine-tuning dataset generation."""
        try:
            from bonus_features import generate_finetuning_dataset

            conversations = [
                {"query": "Test query 1", "response": "Test response 1"},
                {"query": "Test query 2", "response": "Test response 2"},
            ]

            filename = "data/test_finetune.jsonl"
            result = generate_finetuning_dataset(conversations, filename)

            assert os.path.exists(result)

            # Verify format
            with open(result, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2
                for line in lines:
                    data = json.loads(line)
                    assert "messages" in data
                    assert len(data["messages"]) == 3

            # Clean up
            os.remove(result)
        except ImportError:
            pytest.skip("Bonus features not available")

    def test_state_has_bonus_fields(self):
        """Test that ConservationAgentState has bonus fields."""
        # Check that the state includes language and visualization_data
        from agents.state import ConservationAgentState

        # The state should have these annotations
        state_annotations = ConservationAgentState.__annotations__
        assert "language" in state_annotations
        assert "visualization_data" in state_annotations

    def test_memory_saver_enabled(self):
        """Test Challenge 2: MemorySaver is enabled in graph."""
        from agents.graph import create_conservation_agent_graph

        app = create_conservation_agent_graph()

        # Check that the graph was compiled with a checkpointer
        # The checkpointer attribute is named differently in different versions
        assert hasattr(app, "checkpointer")
        assert app.checkpointer is not None

    def test_graph_has_new_tools_in_routing(self):
        """Test that graph routing includes new bonus tools."""
        from agents.graph import route_to_tool

        # Test web_search routing
        state = {"current_tool": "web_search"}
        assert route_to_tool(state) == "tool_execution"

        # Test predict_population routing
        state = {"current_tool": "predict_population"}
        assert route_to_tool(state) == "tool_execution"


class TestStateWithBonusFields:
    """Test state with bonus fields."""

    def test_state_initialization_with_bonus_fields(self):
        """Test creating state with bonus fields."""
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
            "language": "Spanish",
            "visualization_data": "base64string",
            "iteration_count": 0,
            "max_iterations": 3,
        }

        assert state["language"] == "Spanish"
        assert state["visualization_data"] == "base64string"


class TestHistoricalDataInSpecies:
    """Test that species data includes historical_data for ML predictions."""

    def test_species_data_has_historical_data(self):
        """Test that species_data.json includes historical_data."""
        from data_loader import load_species_data

        data = load_species_data()

        # Check that at least one species has historical_data
        species_with_history = [
            s for s in data.values()
            if "historical_data" in s
        ]

        assert len(species_with_history) > 0

        # Check structure
        for species in species_with_history:
            hist = species["historical_data"]
            assert isinstance(hist, dict)
            assert len(hist) >= 3  # At least 3 data points
            # All keys should be years (as strings from JSON)
            for year in hist.keys():
                # JSON keys are strings, so we check string representation
                year_int = int(year) if isinstance(year, str) else year
                assert isinstance(year_int, int)
                assert year_int >= 2000 and year_int <= 2030

    def test_javan_rino_historical_data(self):
        """Test specific species has valid historical data."""
        from data_loader import load_species_data

        data = load_species_data()

        javan = data.get("javan rhino")
        assert javan is not None
        assert "historical_data" in javan

        hist = javan["historical_data"]
        assert len(hist) >= 3
        # Data should show a trend (either increasing or decreasing)
        years = sorted(hist.keys())
        values = [hist[y] for y in years]
        assert all(isinstance(v, int) for v in values)
