#!/usr/bin/env python3
"""Streaming demo script showcasing all bonus features.

This script demonstrates:
- Challenge 7: Streaming Responses
- Challenge 2: Persistent Learning (MemorySaver)
- Challenge 1: Web Search
- Challenge 3: ML Predictions
- Challenge 4: Visualization
- Challenge 5: RAG Integration
- Challenge 6: Fine-tuning Dataset
- Challenge 8: Multi-language Support
"""

import uuid
from main import run_agent_streaming, RAG_AVAILABLE

# If not running from main, we need to initialize RAG
if not RAG_AVAILABLE:
    try:
        from bonus_features import rag_system
        import os

        report_path = "data/animal_conservation_report.md"
        if os.path.exists(report_path):
            rag_system.ingest_report(report_path)
    except ImportError:
        pass


def main():
    """Run the streaming demo with all bonus features."""

    print("\n" + "=" * 70)
    print("  *" + " " * 68 + "*")
    print("  *" + " " * 15 + "LANGRAPH CONSERVATION AGENT" + " " * 26 + "*")
    print("  *" + " " * 20 + "BONUS FEATURES DEMO" + " " * 30 + "*")
    print("  *" + " " * 68 + "*")
    print("=" * 70)

    print("\n🎯 This demo showcases all 8 bonus challenges:")
    print("   1. Web Search Integration")
    print("   2. Persistent Learning (Memory)")
    print("   3. Predictive Analytics (ML)")
    print("   4. Data Visualization")
    print("   5. RAG Integration (Vector Search)")
    print("   6. Fine-tuning Dataset Generation")
    print("   7. Streaming Responses")
    print("   8. Multi-language Support")

    # Create a persistent thread for the session
    thread_id = str(uuid.uuid4())

    print(f"\n📋 Session Thread ID: {thread_id[:8]}... (persistent memory enabled)")
    print("=" * 70)

    # Demo queries for each challenge
    demos = [
        {
            "name": "Challenge 5: RAG Integration",
            "query": "What are the main findings about conservation funding?",
            "description": "Uses vector search to find relevant sections from the report"
        },
        {
            "name": "Challenge 3 & 4: ML Prediction + Visualization",
            "query": "Predict the population trend for Javan Rhino",
            "description": "Uses Linear Regression to forecast population and generates a chart"
        },
        {
            "name": "Challenge 1: Web Search",
            "query": "Search for latest news about elephant conservation",
            "description": "Uses DuckDuckGo search for real-time information"
        },
        {
            "name": "Challenge 8: Multi-language Support",
            "query": "Tell me about giant pandas (respond in Spanish)",
            "description": "Detects language and responds in Spanish"
        },
        {
            "name": "Challenge 2: Persistent Memory",
            "query": "What was the first species I asked about?",
            "description": "Remembers previous conversation from thread"
        },
    ]

    for i, demo in enumerate(demos, 1):
        print(f"\n{'=' * 70}")
        print(f"  DEMO {i}/{len(demos)}: {demo['name']}")
        print(f"{'=' * 70}")
        print(f"  ℹ️  {demo['description']}")
        print(f"  🔍 Query: \"{demo['query']}\"")
        print(f"{'=' * 70}\n")

        try:
            result = run_agent_streaming(
                query=demo['query'],
                thread_id=thread_id
            )

        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  🎉 BONUS FEATURES DEMO COMPLETE!")
    print("=" * 70)
    print("\n📊 Summary of demonstrated features:")
    print("   ✅ Streaming Responses - Watched nodes execute in real-time")
    print("   ✅ Persistent Memory - Thread remembered conversation context")
    print("   ✅ ML Predictions - Linear Regression population forecasts")
    print("   ✅ Visualization - Matplotlib charts generated")
    print("   ✅ RAG Integration - Vector search over conservation report")
    print("   ✅ Web Search - Real-time data from DuckDuckGo")
    print("   ✅ Multi-language - Response language detection")
    print("   ✅ Fine-tuning - Conversations saved to dataset")
    print("\n💾 Fine-tuning dataset saved to: data/finetune_data.jsonl")
    print(f"🧵 Thread ID: {thread_id}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
