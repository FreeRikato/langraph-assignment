"""LangGraph node functions for the conservation agent workflow.

Each node represents a step in the agentic workflow and processes
the state before passing it to the next node.
"""

import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from agents.state import ConservationAgentState
from config import llm
from data_loader import load_conservation_report
from agents.tools import TOOL_REGISTRY

# Bonus features imports
try:
    from bonus_features import rag_system, web_search_tool, predict_and_visualize_population
    BONUS_FEATURES_AVAILABLE = True
except ImportError:
    BONUS_FEATURES_AVAILABLE = False
    rag_system = None


def query_processing_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Parse user query and extract intent, entities, and language.

    Uses the LLM to classify the query intent, extract relevant
    entities such as species names, regions, or taxonomic groups,
    and detect the language for multi-language support.

    Args:
        state: Current agent state.

    Returns:
        Updated state with intent, entities, language, and iteration_count set.
    """
    query = state["query"]
    conversation_history = state.get("conversation_history", [])

    # Build context from conversation history for better entity resolution
    context_info = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 turns for context
        context_info = "\n\nRecent conversation for context:\n" + "\n".join([
            f"Q: {h['query']}\nA: {h['response'][:150]}..."
            for h in recent_history
        ])

    prompt_template = ChatPromptTemplate.from_template(
        """Analyze this conservation query: "{query}"{context_info}

        Classify the intent, extract entities, and detect the language.
        Return ONLY a valid JSON object with this exact format:
        {{
            "intent": "species_info|threat_analysis|funding_analysis|migration_lookup|comparative_analysis|recovery_prediction|general_chat",
            "entities": ["entity1", "entity2", ...],
            "language": "English|Spanish|French|German|Chinese|Japanese|Other"
        }}

        Intent mappings:
        - species_info: Questions about a specific species (population, status, etc.)
        - threat_analysis: Questions about threats to species or regions
        - funding_analysis: Questions about conservation funding, gaps, allocations
        - migration_lookup: Questions about migration patterns
        - comparative_analysis: Questions comparing multiple species
        - recovery_prediction: Questions about recovery chances or interventions
        - general_chat: General conversation, greetings, or unclear queries

        Extract specific species names (e.g., "Javan Rhino", "Giant Panda"), regions, or groups mentioned.
        Use the conversation context to resolve pronouns like "it", "they", "where" to the correct entity.
        Detect the language of the query (default to English if unsure).
        """
    )

    chain = prompt_template | llm

    try:
        response = chain.invoke({"query": query, "context_info": context_info})
        content = response.content.strip()

        # Clean up JSON response (remove markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        intent = parsed.get("intent", "general_chat")
        entities = parsed.get("entities", [])
        language = parsed.get("language", "English")

        # Normalize intent
        valid_intents = [
            "species_info", "threat_analysis", "funding_analysis",
            "migration_lookup", "comparative_analysis", "recovery_prediction", "general_chat"
        ]
        if intent not in valid_intents:
            intent = "general_chat"

        # Add new intents for bonus features
        if "search" in query.lower() or "latest" in query.lower() or "news" in query.lower():
            intent = "web_search"

        return {
            "intent": intent,
            "entities": entities,
            "language": language,
            "iteration_count": state.get("iteration_count", 0)
        }

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error parsing intent: {e}")
        return {
            "intent": "general_chat",
            "entities": [],
            "language": "English",
            "iteration_count": state.get("iteration_count", 0)
        }


def data_retrieval_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Retrieve relevant data from the conservation report.

    Uses RAG (Retrieval-Augmented Generation) with vector search
    when available. Falls back to full document loading.

    Args:
        state: Current agent state.

    Returns:
        Updated state with retrieved_data populated.
    """
    query = state.get("query", "")

    try:
        # Try RAG first (Challenge 5)
        if BONUS_FEATURES_AVAILABLE and rag_system and rag_system.vector_store:
            print("[Data Retrieval] Using RAG vector search")
            context = rag_system.query(query, k=3)

            return {
                "retrieved_data": {
                    "rag_context": context,
                    "source": "vector_search",
                    "sections": {
                        "executive_summary": True,
                        "funding_analysis": True,
                        "threat_analysis": True,
                        "migration_patterns": True,
                    }
                }
            }
        else:
            # Fallback to full document loading
            print("[Data Retrieval] Using full document load")
            report_text = load_conservation_report()

            # Truncate to avoid token overflow
            if len(report_text) > 5000:
                report_text = report_text[:5000] + "..."

            return {
                "retrieved_data": {
                    "report_context": report_text,
                    "source": "full_document",
                    "sections": {
                        "executive_summary": True,
                        "funding_analysis": True,
                        "threat_analysis": True,
                        "migration_patterns": True,
                    }
                }
            }
    except Exception as e:
        print(f"Error loading conservation report: {e}")
        return {
            "retrieved_data": {"report_context": "", "error": str(e), "source": "error"}
        }


def tool_selection_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Select the appropriate tool based on query intent.

    Maps the classified intent to the corresponding analysis tool.
    Includes bonus features: web_search, predict_population.

    Args:
        state: Current agent state.

    Returns:
        Updated state with current_tool set.
    """
    intent = state.get("intent", "general_chat")

    # Map intents to tools (including bonus features)
    intent_to_tool = {
        "species_info": "retrieve_species_info",
        "threat_analysis": "analyze_threats",
        "funding_analysis": "analyze_funding",
        "migration_lookup": "get_migration_data",
        "comparative_analysis": "compare_conservation_metrics",
        "recovery_prediction": "predict_recovery_success",
        "web_search": "web_search",  # Challenge 1
        "predict_population": "predict_population",  # Challenge 3
    }

    current_tool = intent_to_tool.get(intent, "none")

    # Detect prediction intent from keywords
    query = state.get("query", "").lower()
    if "predict" in query and "population" in query:
        current_tool = "predict_population"

    print(f"[Tool Selection] Intent: {intent} -> Tool: {current_tool}")

    return {"current_tool": current_tool}


def tool_execution_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Execute the selected tool with appropriate parameters.

    Handles bonus features: web_search, predict_population with visualization.

    Args:
        state: Current agent state.

    Returns:
        Updated state with tool_results and visualization_data populated.
    """
    tool_name = state.get("current_tool", "none")
    entities = state.get("entities", [])
    intent = state.get("intent", "general_chat")
    query = state.get("query", "")

    print(f"[Tool Execution] Executing: {tool_name}")

    results = {}
    visualization_data = state.get("visualization_data", "")

    try:
        # Handle bonus feature tools first
        if tool_name == "web_search" and BONUS_FEATURES_AVAILABLE:
            # Challenge 1: Web Search
            search_query = entities[0] if entities else query
            results = web_search_tool(search_query, max_results=3)

        elif tool_name == "predict_population" and BONUS_FEATURES_AVAILABLE:
            # Challenge 3 & 4: ML Prediction + Visualization
            from data_loader import load_species_data

            species_name = entities[0] if entities else "Javan Rhino"
            data = load_species_data()

            # Get historical data for the species (or use mock data)
            for key, species_data in data.items():
                if species_name.lower() in key or key in species_name.lower():
                    historical = species_data.get("historical_data", {2010: 80, 2015: 78, 2020: 76})
                    results = predict_and_visualize_population(species_name, historical)
                    visualization_data = results.get("chart_base64", "")
                    break
            else:
                # Mock data for demo
                mock_history = {2010: 85, 2015: 80, 2020: 76}
                results = predict_and_visualize_population(species_name, mock_history)
                visualization_data = results.get("chart_base64", "")

        elif tool_name == "none" or tool_name not in TOOL_REGISTRY:
            results = {"error": "No tool selected or tool not found"}

        else:
            # Standard tools from TOOL_REGISTRY
            tool_func = TOOL_REGISTRY[tool_name]

            # Prepare arguments based on tool type
            if tool_name == "retrieve_species_info":
                arg = entities[0] if entities else ""
                results = tool_func(arg)

            elif tool_name == "analyze_threats":
                arg = entities[0] if entities else ""
                results = tool_func(arg)

            elif tool_name == "analyze_funding":
                arg = entities[0] if entities else ""
                results = tool_func(arg)

            elif tool_name == "get_migration_data":
                arg = entities[0] if entities else ""
                results = tool_func(arg)

            elif tool_name == "compare_conservation_metrics":
                # For comparison, use all entities
                metric = "population"  # Default metric
                results = tool_func(entities, metric)

            elif tool_name == "predict_recovery_success":
                arg = entities[0] if entities else ""
                intervention = "habitat restoration"  # Default intervention
                results = tool_func(arg, intervention)

    except Exception as e:
        print(f"[Tool Execution] Error: {e}")
        results = {"error": str(e), "tool": tool_name}

    return {
        "tool_results": results,
        "visualization_data": visualization_data
    }


def analysis_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Synthesize tool results and retrieved data to generate insights.

    Uses the LLM to analyze the data and generate meaningful insights.
    Handles both RAG context and traditional document context.

    Args:
        state: Current agent state.

    Returns:
        Updated state with analysis_result populated.
    """
    tool_results = state.get("tool_results", {})
    retrieved_data = state.get("retrieved_data", {})
    query = state["query"]

    # Handle both RAG and traditional context
    if "rag_context" in retrieved_data:
        report_context = retrieved_data.get("rag_context", "")
        source = "RAG (vector search)"
    else:
        report_context = retrieved_data.get("report_context", "")
        source = "document"

    # Truncate context to avoid token overflow
    if len(report_context) > 3000:
        report_context = report_context[:3000] + "..."

    prompt = f"""You are a conservation science expert. Analyze the following data to answer the user's query.

User Query: {query}

Tool Results: {json.dumps(tool_results, indent=2, default=str)}

Conservation Report Context (from {source}): {report_context}

Provide a comprehensive, scientific analysis that:
1. Directly answers the user's question
2. Includes relevant data and statistics
3. Provides context and implications
4. Suggests conservation actions when relevant

Be thorough but concise. Focus on actionable insights.
"""

    try:
        response = llm.invoke(prompt)
        analysis = response.content
    except Exception as e:
        print(f"[Analysis] Error: {e}")
        analysis = f"Analysis error: {str(e)}"

    return {"analysis_result": analysis}


def response_generation_node(state: ConservationAgentState) -> ConservationAgentState:
    """
    Format the final response for the user.

    Supports multi-language responses and includes visualization data
    when available (Challenge 4 & 8).

    Args:
        state: Current agent state.

    Returns:
        Updated state with final_response populated.
    """
    analysis = state.get("analysis_result", "")
    language = state.get("language", "English")
    visualization_data = state.get("visualization_data", "")
    conversation_history = state.get("conversation_history", [])

    # Build conversation context for response generation
    conversation_context = ""
    if conversation_history:
        recent_history = conversation_history[-5:]  # Last 5 turns
        conversation_context = "\n\nPrevious conversation:\n" + "\n".join([
            f"Q: {h['query']}\nA: {h['response'][:200]}..."
            for h in recent_history
        ])

    # Build visualization indicator
    viz_note = ""
    if visualization_data:
        viz_note = "\n\n[📊 A population forecast chart has been generated for this query.]"

    prompt = f"""Format the following conservation analysis into a professional, user-friendly response.

Analysis: {analysis}{conversation_context}

Target Language: {language} (Translate your response to this language if it's not English)

Guidelines:
- Use clear headings and structure
- Highlight key statistics and findings
- Maintain a professional but accessible tone
- Include actionable recommendations when relevant
- Use bullet points for lists
- If the language is not English, provide the full response in that language{viz_note}
- IMPORTANT: If the user shared personal information (name, preferences) in previous conversation, remember and acknowledge it appropriately

Output the formatted response directly, without introductory text.
"""

    try:
        response = llm.invoke(prompt)
        final = response.content
    except Exception as e:
        print(f"[Response Generation] Error: {e}")
        final = analysis  # Fallback to raw analysis

    # Append visualization note if chart was generated
    if visualization_data and "[📊" not in final:
        final += f"\n\n[📊 A population forecast chart has been generated (Base64 length: {len(visualization_data)} chars)]"

    # Update conversation history
    history_entry = {
        "query": state["query"],
        "response": final,
        "language": language
    }

    updated_history = state.get("conversation_history", []) + [history_entry]

    return {
        "final_response": final,
        "conversation_history": updated_history
    }
