### Phase 1: Foundation & Setup (Part 1 of Docs)
*   [x] **1.1 Project Initialization**
    *   [x] Initialize `uv` project: `uv init conservation-agent`
    *   [x] Create directory structure:
        *   `agents/`
        *   `data/`
        *   `tests/`
    *   [x] Create `.env` file and add `GROQ_API_KEY`.
*   [x] **1.2 Dependencies**
    *   [x] Install core packages: `uv add langgraph langchain langchain-groq python-dotenv pydantic pandas`
    *   [x] Install dev packages: `uv add --dev pytest`
*   [x] **1.3 Data Preparation**
    *   [x] Create `data/species_data.json` with mock data (Rhinos, Elephants, Pandas, + 7 more species).
    *   [x] Create `data/animal_conservation_report.md` with mock executive summary text.
    *   [x] Create `data_loader.py` helper to read these files.

### Phase 2: Agent Architecture Design (Part 2 of Docs)
*   [x] **2.1 Define Agent State**
    *   [x] Create `agents/state.py`.
    *   [x] Define `ConservationAgentState(TypedDict)`.
    *   [x] Include required keys: `query`, `conversation_history`, `current_tool`, `retrieved_data`, `analysis_result`, `tool_results`.
*   [x] **2.2 Configuration**
    *   [x] Create `config.py`.
    *   [x] Initialize `ChatGroq` model (e.g., `llama3-70b-8192`) here.

### Phase 3: Implement Tools (Part 3 of Docs)
*   [x] **3.1 Species Information Retrieval**
    *   [x] Implement `retrieve_species_info(species_name: str)`.
    *   [x] Add fuzzy matching logic (handle "rhino" matching "Javan Rhino").
*   [x] **3.2 Threat Analysis**
    *   [x] Implement `analyze_threats(target: str, threat_type: str)`.
    *   [x] Logic to return severity scores.
*   [x] **3.3 Funding Analysis**
    *   [x] Implement `analyze_funding(category: str)`.
    *   [x] Logic to calculate gaps/allocation percentages.
*   [x] **3.4 Migration Pattern Lookup**
    *   [x] Implement `get_migration_data(species_name: str)`.
    *   [x] Return routes and seasonal info.
*   [x] **3.5 Comparative Analysis**
    *   [x] Implement `compare_conservation_metrics(entities: list, metric: str)`.
    *   [x] Logic to return a comparison dictionary/table.
*   [x] **3.6 Recovery Success Prediction**
    *   [x] Implement `predict_recovery_success(species_name: str, intervention: str)`.
    *   [x] Logic to return probability score.

### Phase 4: Implement Node Logic (Part 5 of Docs)
*   [x] **4.1 Query Processing Node**
    *   [x] Create prompt to extract `intent` and `entities` using Groq.
    *   [x] Output JSON structure.
*   [x] **4.2 Data Retrieval Node**
    *   [x] Logic to read `animal_conservation_report.md` into state.
*   [x] **4.3 Tool Selection Node**
    *   [x] LLM logic to look at `intent` and map to one of the 6 function names.
    *   [x] Handle "General Chat" (no tool) scenario.
*   [x] **4.4 Tool Execution Node**
    *   [x] Create mapping dictionary `{ "tool_name": function_reference }`.
    *   [x] Execute function with arguments from state.
    *   [x] Implement error handling (try/except) if tool fails.
*   [x] **4.5 Analysis & Reasoning Node**
    *   [x] Create prompt to synthesize `tool_results` + `retrieved_data`.
    *   [x] Generate insights based on the query.
*   [x] **4.6 Response Generation Node**
    *   [x] Create prompt to format the final answer politely.

### Phase 5: Build LangGraph Workflow (Part 4 of Docs)
*   [x] **5.1 Graph Definition**
    *   [x] Initialize `StateGraph(ConservationAgentState)`.
    *   [x] Add all 6 nodes to the graph.
*   [x] **5.2 Edge Logic**
    *   [x] Add standard edges (Query -> Retrieval -> Selection).
    *   [x] Implement Conditional Edge: `route_to_tool` (Selection -> Execution OR Analysis).
    *   [x] Implement Loop Logic (Optional/Bonus): Allow Analysis to loop back to Tool Selection if data is missing.
*   [x] **5.3 Compilation**
    *   [x] Compile graph with `app = graph.compile()`.

### Phase 6: Testing & Validation (Part 6 of Docs)
*   [x] **6.1 Unit Tests**
    *   [x] Create `tests/test_nodes.py`.
    *   [x] Test `query_processing` (does it extract "Rhino"?).
    *   [x] Test `tool_execution` (does it return JSON?).
*   [x] **6.2 Integration Scenarios**
    *   [x] Create `tests/test_workflow.py` or a `manual_test.py` script.
    *   [x] **Test Case A:** "Tell me about the Javan Rhino" (Simple lookup).
    *   [x] **Test Case B:** "What are the threats to African Elephants?" (Threat tool).
    *   [x] **Test Case C:** "Which species is underfunded?" (Funding tool).
    *   [x] **Test Case D:** "Compare funding for Rhinos vs Pandas" (Comparative tool).

### Phase 7: Documentation & Polish (Part 7 of Docs)
*   [x] **7.1 README**
    *   [x] Write setup instructions (using `uv`).
    *   [x] Include Architecture Diagram (Mermaid or text description).
*   [x] **7.2 Code Quality**
    *   [x] Add Type Hints to all functions.
    *   [x] Add Docstrings to nodes and tools.
*   [x] **7.3 Final Submission Check**
    *   [x] Check against "Deliverables Checklist" in the assignment image.

---

## Summary
**All tasks completed!** ✅

- **19/19 tests passing**
- **6 tools implemented**
- **6 nodes implemented**
- **Full LangGraph workflow with conditional routing**
- **Complete documentation with README**
