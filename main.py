"""Main entry point for the conservation agent.

Provides a command-line interface for running the agent and
includes built-in test cases.

Bonus Features:
- RAG Integration (Challenge 5)
- Streaming Responses (Challenge 7)
- Persistent Memory with thread_id (Challenge 2)
"""

import uuid
from typing import Dict, Any

from agents.graph import app
from agents.state import ConservationAgentState

# Initialize RAG system (Challenge 5)
try:
    from bonus_features import rag_system, append_to_finetuning_dataset
    import os

    # Ingest the conservation report into the vector store
    report_path = "data/animal_conservation_report.md"
    if os.path.exists(report_path):
        rag_system.ingest_report(report_path)
        print("✅ RAG System initialized with conservation report")
    else:
        print(f"⚠️ Report file not found: {report_path}")
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ Bonus features not available")


def run_agent(
    query: str,
    verbose: bool = True,
    thread_id: str | None = None,
    language: str = "English"
) -> Dict[str, Any]:
    """
    Run the conservation agent with a query.

    Args:
        query: The user's question about animal conservation.
        verbose: Whether to print execution details.
        thread_id: Optional thread ID for persistent memory (Challenge 2).
        language: Target language for the response (Challenge 8).

    Returns:
        The final state after processing the query.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        if thread_id:
            print(f"Thread: {thread_id[:8]}... (persistent memory)")
        print("=" * 60)

    # Initialize the state with bonus fields
    # Note: conversation_history is omitted - let checkpointer load and accumulate it
    initial_state: ConservationAgentState = {
        "query": query,
        "intent": "",
        "entities": [],
        "retrieved_data": {},
        "current_tool": "",
        "tool_results": {},
        "analysis_result": "",
        "final_response": "",
        "language": language,  # Challenge 8
        "visualization_data": "",  # Challenge 4
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Run the graph with optional thread_id for persistent memory
    # MemorySaver checkpointer requires thread_id, so generate one if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config=config)

    # Append to fine-tuning dataset (Challenge 6)
    if RAG_AVAILABLE and result.get("final_response"):
        append_to_finetuning_dataset(query, result["final_response"])

    if verbose:
        print("\n--- Response ---")
        print(result.get("final_response", "No response generated."))
        print("=" * 60 + "\n")

    return result


def run_agent_streaming(
    query: str,
    thread_id: str | None = None,
    language: str = "English"
) -> Dict[str, Any]:
    """
    Run the conservation agent with streaming responses (Challenge 7).

    Args:
        query: The user's question about animal conservation.
        thread_id: Optional thread ID for persistent memory.
        language: Target language for the response.

    Returns:
        The final state after processing the query.
    """
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    if thread_id:
        print(f"Thread: {thread_id[:8]}... (persistent memory)")
    print("=" * 60 + "\n")

    # Initialize the state
    # Note: conversation_history is omitted - let checkpointer load and accumulate it
    initial_state: ConservationAgentState = {
        "query": query,
        "intent": "",
        "entities": [],
        "retrieved_data": {},
        "current_tool": "",
        "tool_results": {},
        "analysis_result": "",
        "final_response": "",
        "language": language,
        "visualization_data": "",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # Stream the graph execution
    print("🔄 Agent execution starting...\n")

    for output in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_value in output.items():
            print(f"✓ Node completed: {node_name}")

            # Print specific outputs
            if "intent" in node_value:
                print(f"  └─ Intent: {node_value['intent']}")
            if "current_tool" in node_value and node_value["current_tool"]:
                print(f"  └─ Tool: {node_value['current_tool']}")
            if "visualization_data" in node_value and node_value["visualization_data"]:
                print(f"  └─ 📊 Chart generated ({len(node_value['visualization_data'])} bytes)")
            if "final_response" in node_value and node_value["final_response"]:
                print(f"\n🤖 Response:\n{node_value['final_response']}")

    # Get final state
    final_state = app.get_state(config).values

    # Append to fine-tuning dataset
    if RAG_AVAILABLE and final_state.get("final_response"):
        append_to_finetuning_dataset(query, final_state["final_response"])

    print("\n" + "=" * 60 + "\n")

    return final_state


def run_test_cases():
    """Run the built-in test cases from TASKS.md."""
    test_cases = [
        ("Test Case A - Species Info", "Tell me about the Javan Rhino"),
        ("Test Case B - Threat Analysis", "What are the main threats to African Elephants?"),
        ("Test Case C - Funding Analysis", "Which animal groups receive the least funding?"),
        ("Test Case D - Comparative Analysis", "Compare funding for Rhinos vs Pandas"),
    ]

    results = []

    for name, query in test_cases:
        print(f"\n{'#' * 60}")
        print(f"# {name}")
        print(f"{'#' * 60}")

        try:
            result = run_agent(query, verbose=True)
            results.append((name, "SUCCESS", None))
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            results.append((name, "FAILED", str(e)))

    # Summary
    print(f"\n{'#' * 60}")
    print("# Test Summary")
    print(f"{'#' * 60}")

    passed = sum(1 for _, status, _ in results if status == "SUCCESS")
    total = len(results)

    for name, status, error in results:
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {name}: {status}")
        if error:
            print(f"  Error: {error}")

    print(f"\n{passed}/{total} tests passed")

    return results


def run_bonus_features_demo():
    """Demonstrate all bonus features (Challenges 1-8)."""
    print("\n" + "=" * 60)
    print("  BONUS FEATURES DEMO")
    print("=" * 60 + "\n")

    # Create a persistent thread for the session
    thread_id = str(uuid.uuid4())
    print(f"Session Thread ID: {thread_id[:8]}...\n")

    demo_queries = [
        ("Challenge 1 - Web Search", "Search for latest news about rhino conservation"),
        ("Challenge 3 - ML Prediction", "Predict the population trend for Javan Rhino"),
        ("Challenge 8 - Multi-language", "Tell me about pandas (respond in Spanish)"),
    ]

    for name, query in demo_queries:
        print(f"\n{'#' * 60}")
        print(f"# {name}")
        print(f"{'#' * 60}")
        print(f"Query: {query}\n")

        try:
            result = run_agent(query, verbose=False, thread_id=thread_id)
            print(result.get("final_response", "No response generated."))
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Bonus features demo complete!")
    print(f"💾 Conversation saved to fine-tuning dataset")
    print("=" * 60 + "\n")


def interactive_mode():
    """Run the agent in interactive mode with persistent memory."""
    print("\n" + "=" * 60)
    print("  Conservation Agent - Interactive Mode")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    # Create persistent thread for the session
    thread_id = str(uuid.uuid4())
    print(f"Session Thread ID: {thread_id[:8]}... (your conversation will be remembered)\n")

    conversation_history = []

    while True:
        try:
            query = input("Your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!\n")
                break

            if not query:
                continue

            result = run_agent(query, verbose=True, thread_id=thread_id)
            conversation_history.append({
                "query": query,
                "response": result.get("final_response", "")
            })

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "test":
            # Run test cases
            run_test_cases()

        elif command == "interactive" or command == "i":
            # Interactive mode
            interactive_mode()

        elif command == "streaming" or command == "s":
            # Streaming mode demo
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Tell me about the Javan Rhino"
            run_agent_streaming(query)

        elif command == "bonus":
            # Bonus features demo
            run_bonus_features_demo()

        elif command == "query" and len(sys.argv) > 2:
            # Single query mode
            query = " ".join(sys.argv[2:])
            run_agent(query, verbose=True)

        else:
            print("Usage:")
            print("  uv run main.py test          - Run test cases")
            print("  uv run main.py interactive   - Interactive mode (with memory)")
            print("  uv run main.py streaming     - Streaming mode demo")
            print("  uv run main.py bonus        - Bonus features demo")
            print("  uv run main.py query <text>  - Ask a single question")
            print("\nRunning test cases by default...")
            run_test_cases()
    else:
        # Default: run test cases
        print("No command specified. Running test cases...\n")
        run_test_cases()


if __name__ == "__main__":
    main()
