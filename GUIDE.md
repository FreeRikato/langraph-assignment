# LangGraph Conservation Agent - Study Guide

> A comprehensive guide for understanding and revising the LangGraph Animal Conservation Agent implementation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [State Management](#state-management)
4. [Node Functions](#node-functions)
5. [Graph Workflow](#graph-workflow)
6. [Analysis Tools](#analysis-tools)
7. [Bonus Features](#bonus-features)
8. [Running the Agent](#running-the-agent)
9. [Testing](#testing)
10. [Key Concepts](#key-concepts)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query                                                          │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ Query Processing│  (Extract intent, entities, language)          │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  Data Retrieval │  (RAG vector search or document load)          │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  Tool Selection │  (Map intent to appropriate tool)              │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ├─► (if tool selected) ──► ┌──────────────┐              │
│           │                          │Tool Execution│              │
│           │                          └──────┬───────┘              │
│           │                                 │                       │
│           └─────────────────────────────────┘                       │
│                     │                                                │
│                     ▼                                                │
│           ┌─────────────┐                                          │
│           │   Analysis  │  (LLM synthesizes insights)               │
│           └──────┬──────┘                                          │
│                  │                                                 │
│                  ├─► (needs more data) ──► Tool Selection (loop)   │
│                  │                                                 │
│                  ▼                                                 │
│           ┌─────────────┐                                          │
│           │  Response   │  (Format for user, multi-language)       │
│           │  Generation │                                          │
│           └──────┬──────┘                                          │
│                  │                                                 │
│                  ▼                                                 │
│           Final Response                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
missoula/
├── main.py                 # Entry point with CLI, streaming, RAG init
├── config.py               # Groq LLM configuration
├── data_loader.py          # Data loading utilities
├── bonus_features.py       # Bonus challenges 1-8
├── run_bonus.py            # Standalone bonus demo
│
├── agents/
│   ├── __init__.py
│   ├── state.py            # ConservationAgentState TypedDict
│   ├── tools.py            # 6 analysis tools + TOOL_REGISTRY
│   ├── nodes.py            # 6 LangGraph node functions
│   └── graph.py            # StateGraph definition + routing
│
├── data/
│   ├── species_data.json   # 10 species with historical data
│   └── animal_conservation_report.md  # RAG document source
│
├── tests/
│   ├── test_nodes.py       # Node unit tests (13 tests)
│   ├── test_workflow.py    # Integration tests (7 tests)
│   └── test_bonus.py       # Bonus feature tests (12 tests)
│
├── TASKS.md                # Assignment checklist
├── README.md               # Setup instructions
└── GUIDE.md                # This file
```

---

## State Management

### ConservationAgentState (`agents/state.py`)

The state is a **TypedDict** that flows through all nodes in the LangGraph:

```python
class ConservationAgentState(TypedDict):
    # Input fields
    query: str                           # User's question

    # Conversation tracking
    conversation_history: Annotated[List[Dict[str, Any]], operator.add]

    # Query analysis (populated by query_processing_node)
    intent: str                          # species_info, threat_analysis, etc.
    entities: List[str]                  # Extracted species names, regions

    # Data fields (populated by data_retrieval_node)
    retrieved_data: Dict[str, Any]       # Content from report

    # Tool execution (populated by tool_selection_node, tool_execution_node)
    current_tool: str                    # Selected tool name
    tool_results: Dict[str, Any]         # Tool output

    # Analysis (populated by analysis_node)
    analysis_result: str                 # LLM's analysis

    # Output (populated by response_generation_node)
    final_response: str                  # Formatted response

    # Bonus fields (Challenges 4, 8)
    language: str                        # Target language (default: "English")
    visualization_data: str              # Base64 encoded chart

    # Flow control
    iteration_count: int                 # Prevent infinite loops
    max_iterations: int                  # Max loops (default: 3)
```

**Key Design Decisions:**
- Uses `operator.add` annotation for `conversation_history` to automatically append
- All fields are optional updates (nodes return dict with partial state updates)
- State is immutable between nodes (LangGraph handles merging)

---

## Node Functions

### 1. Query Processing Node (`agents/nodes.py:26-110`)

**Purpose:** Transform natural language into structured query parameters.

**LLM Prompt Strategy:**
```python
prompt = """Analyze this conservation query: "{query}"
Return JSON with:
- intent: species_info|threat_analysis|funding_analysis|...
- entities: ["entity1", "entity2", ...]
- language: English|Spanish|French|...
"""
```

**Processing Steps:**
1. Send query to LLM with structured prompt
2. Parse JSON response (handle markdown code blocks)
3. Validate intent against allowed values
4. Detect web_search intent for keywords ("latest", "news", "search")
5. Return partial state update: `{intent, entities, language, iteration_count}`

**Error Handling:** Falls back to `general_chat` intent if JSON parsing fails.

---

### 2. Data Retrieval Node (`agents/nodes.py:113-171`)

**Purpose:** Fetch relevant information from the conservation report.

**Two Paths:**

| Path | Condition | Source |
|------|-----------|--------|
| RAG | `rag_system.vector_store` exists | ChromaDB vector search |
| Fallback | RAG unavailable | Full document load |

**RAG Implementation:**
```python
if BONUS_FEATURES_AVAILABLE and rag_system and rag_system.vector_store:
    context = rag_system.query(query, k=3)  # Top 3 matches
    return {"retrieved_data": {"rag_context": context, "source": "vector_search"}}
```

**Fallback:**
```python
report_text = load_conservation_report()
if len(report_text) > 5000:
    report_text = report_text[:5000] + "..."  # Prevent token overflow
```

---

### 3. Tool Selection Node (`agents/nodes.py:174-210`)

**Purpose:** Map query intent to the appropriate analysis tool.

**Intent-to-Tool Mapping:**

```python
intent_to_tool = {
    "species_info": "retrieve_species_info",
    "threat_analysis": "analyze_threats",
    "funding_analysis": "analyze_funding",
    "migration_lookup": "get_migration_data",
    "comparative_analysis": "compare_conservation_metrics",
    "recovery_prediction": "predict_recovery_success",
    "web_search": "web_search",           # Bonus Challenge 1
    "predict_population": "predict_population",  # Bonus Challenge 3
}
```

**Special Logic:**
- Detects "predict" + "population" keywords for ML prediction
- Returns `"none"` for general_chat (skips tool execution)

---

### 4. Tool Execution Node (`agents/nodes.py:213-303`)

**Purpose:** Execute the selected tool and capture results.

**Execution Flow:**

```
┌─────────────────────────────────────────────────────┐
│ 1. Check if bonus feature tool                      │
│    ├─ web_search → call web_search_tool()          │
│    └─ predict_population → ML prediction + chart   │
│                                                      │
│ 2. Else, use TOOL_REGISTRY                          │
│    ├─ retrieve_species_info → single entity arg    │
│    ├─ analyze_threats → entity + threat_type       │
│    ├─ analyze_funding → category argument          │
│    ├─ get_migration_data → species_name            │
│    ├─ compare_conservation_metrics → entities list │
│    └─ predict_recovery_success → species + intervention │
│                                                      │
│ 3. Handle errors gracefully                         │
│ 4. Store visualization_data if chart generated      │
└─────────────────────────────────────────────────────┘
```

**Bonus Tool Handling:**
```python
if tool_name == "predict_population" and BONUS_FEATURES_AVAILABLE:
    historical = species_data.get("historical_data", {2010: 80, 2015: 78, 2020: 76})
    results = predict_and_visualize_population(species_name, historical)
    visualization_data = results.get("chart_base64", "")  # Store for response
```

---

### 5. Analysis Node (`agents/nodes.py:306-359`)

**Purpose:** Use LLM to synthesize tool results and context into insights.

**Prompt Construction:**
```python
prompt = f"""You are a conservation science expert.

User Query: {query}
Tool Results: {json.dumps(tool_results)}
Report Context: {report_context}

Provide analysis that:
1. Directly answers the question
2. Includes data and statistics
3. Provides context and implications
4. Suggests conservation actions
"""
```

**Key Features:**
- Handles both RAG and traditional document context
- Truncates context to prevent token overflow
- Falls back to raw analysis string if LLM fails

---

### 6. Response Generation Node (`agents/nodes.py:362-424`)

**Purpose:** Format the analysis into a user-friendly response.

**Features:**
- Multi-language support (Challenge 8)
- Visualization embedding (Challenge 4)
- Conversation history tracking

```python
prompt = f"""Format the analysis for the user.

Target Language: {language} (Translate if not English)

Guidelines:
- Clear headings and structure
- Highlight key statistics
- Professional but accessible tone
- Bullet points for lists
{viz_note}  # Add chart note if generated
"""
```

---

## Graph Workflow

### Graph Definition (`agents/graph.py:79-144`)

```python
def create_conservation_agent_graph():
    workflow = StateGraph(ConservationAgentState)

    # Add nodes
    workflow.add_node("query_processing", query_processing_node)
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("tool_selection", tool_selection_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("response_generation", response_generation_node)

    # Entry point
    workflow.set_entry_point("query_processing")

    # Edges (linear)
    workflow.add_edge("query_processing", "data_retrieval")
    workflow.add_edge("data_retrieval", "tool_selection")
    workflow.add_edge("tool_execution", "analysis")
    workflow.add_edge("response_generation", END)

    # Conditional edges
    workflow.add_conditional_edges("tool_selection", route_to_tool, {...})
    workflow.add_conditional_edges("analysis", check_iteration_limit, {...})

    # Compile with MemorySaver (Challenge 2)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
```

### Conditional Routing

**`route_to_tool` (lines 23-50):**
```python
def route_to_tool(state: ConservationAgentState) -> str:
    current_tool = state.get("current_tool", "none")
    valid_tools = [..., "web_search", "predict_population"]

    if current_tool == "none" or current_tool not in valid_tools:
        return "analysis"      # Skip tool execution
    return "tool_execution"   # Execute tool
```

**`check_iteration_limit` (lines 53-76):**
```python
def check_iteration_limit(state: ConservationAgentState) -> str:
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "response_generation"  # Prevent infinite loops

    # Check if analysis needs more data
    if "more information" in analysis.lower():
        return "tool_selection"       # Loop back

    return "response_generation"
```

---

## Analysis Tools

### Tool Registry (`agents/tools.py:397-405`)

```python
TOOL_REGISTRY = {
    "retrieve_species_info": ConservationTools.retrieve_species_info,
    "analyze_threats": ConservationTools.analyze_threats,
    "analyze_funding": ConservationTools.analyze_funding,
    "get_migration_data": ConservationTools.get_migration_data,
    "compare_conservation_metrics": ConservationTools.compare_conservation_metrics,
    "predict_recovery_success": ConservationTools.predict_recovery_success,
}
```

### Tool 1: Species Info (`tools.py:17-50`)

**Signature:** `retrieve_species_info(species_name: str) -> dict`

**Returns:**
- `scientific_name`, `population`, `status`, `location`
- `threats` (list), `migration` description
- `funding_received`, `funding_needed`
- `population_trend`, `recovery_probability`
- `matched_name` (fuzzy match key)

**Fuzzy Matching:** Uses `fuzzy_match_species()` from `data_loader.py`

---

### Tool 2: Threat Analysis (`tools.py:53-130`)

**Signature:** `analyze_threats(target: str, threat_type: str = "") -> dict`

**Severity Calculation:**
```python
if status == "Critically Endangered" or population < 100:
    severity = "Critical"
elif status == "Endangered" or population < 1000:
    severity = "High"
...
```

**Mitigation Recommendations:** Generated based on threat keywords (poaching, habitat loss, climate change).

---

### Tool 3: Funding Analysis (`tools.py:133-201`)

**Signature:** `analyze_funding(category: str) -> dict`

**Two Modes:**
1. **Taxonomic group** (mammals, birds, etc.) - returns allocation percentage
2. **Specific species** - returns received/needed/gap calculations

**Status Logic:**
```python
percentage = (received / needed * 100)
if percentage >= 80: status = "Well funded"
elif percentage >= 50: status = "Adequately funded"
else: status = "Underfunded"
```

---

### Tool 4: Migration Data (`tools.py:204-247`)

**Signature:** `get_migration_data(species_name: str) -> dict`

**Returns:**
- `migration_pattern` description
- `location`
- `conservation_challenges` (based on migration type)

---

### Tool 5: Comparative Metrics (`tools.py:250-307`)

**Signature:** `compare_conservation_metrics(entities: list[str], metric: str) -> dict`

**Supported Metrics:**
- `population` - raw count
- `funding` - funding_received
- `threats` - count of threats
- `recovery_probability` - probability score
- `status` - conservation status

**Output:** Ranked comparison from best to worst

---

### Tool 6: Recovery Prediction (`tools.py:310-394`)

**Signature:** `predict_recovery_success(species_name: str, intervention: str) -> dict`

**Intervention Multipliers:**
```python
intervention_effectiveness = {
    "legal protection": 1.35,
    "habitat restoration": 1.30,
    "captive breeding": 1.25,
    "anti poaching": 1.20,
    ...
}
projected = min(0.95, baseline * multiplier)
```

**Timeline Estimation:** Based on population size (<100 → 50-100 years, etc.)

---

## Bonus Features

### Challenge 1: Web Search (`bonus_features.py:31-50`)

```python
def web_search_tool(query: str, max_results: int = 3) -> List[Dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results  # [{title, url, body}, ...]
```

**Integration:** Triggered by "search", "latest", "news" keywords.

---

### Challenge 2: Persistent Learning (`agents/graph.py:141-142`)

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

**Usage:** Pass `thread_id` in config to maintain conversation memory:
```python
config = {"configurable": {"thread_id": thread_id}}
result = app.invoke(state, config=config)
```

---

### Challenge 3: Predictive Analytics (`bonus_features.py:57-128`)

```python
def predict_and_visualize_population(species_name: str, historical_data: Dict[int, int]):
    # Prepare data
    years = np.array(list(historical_data.keys())).reshape(-1, 1)
    counts = np.array(list(historical_data.values()))

    # Train Linear Regression
    model = LinearRegression()
    model.fit(years, counts)

    # Predict next 5 years
    future_years = ...
    predictions = model.predict(future_years)

    return {
        "forecast": {...},
        "trend": "Increasing/Decreasing/Stable",
        "r_squared": model.score(years, counts)
    }
```

---

### Challenge 4: Visualization (`bonus_features.py:96-119`)

```python
plt.figure(figsize=(10, 6))
plt.scatter(years, counts, label='Historical Data')
plt.plot(future_years, predictions, label='ML Prediction')

# Save to Base64
buf = io.BytesIO()
plt.savefig(buf, format='png')
img_str = base64.b64encode(buf.read()).decode('utf-8')

return {"chart_base64": img_str}
```

**Display:** Embedded in `final_response` as note with byte length.

---

### Challenge 5: RAG Integration (`bonus_features.py:135-243`)

```python
class ConservationRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def ingest_report(self, file_path: str):
        # Split by ## headers
        sections = text.split("##")
        docs = [Document(page_content=s, metadata={...}) for s in sections]

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_fn,
            persist_directory=self.persist_directory
        )

    def query(self, query: str, k: int = 3) -> str:
        results = self.vector_store.similarity_search(query, k=k)
        return "\n\n---\n\n".join([doc.page_content for doc in results])
```

**Initialization:** In `main.py:19-29`

---

### Challenge 6: Fine-tuning Dataset (`bonus_features.py:250-306`)

```python
def generate_finetuning_dataset(conversations: List[Dict], filename: str):
    # Llama-3/Chat format
    for convo in conversations:
        entry = {
            "messages": [
                {"role": "system", "content": "You are a conservation expert..."},
                {"role": "user", "content": convo['query']},
                {"role": "assistant", "content": convo['response']}
            ]
        }
        f.write(json.dumps(entry) + "\n")  # JSONL format
```

**Auto-collection:** Every query/response appended to `data/finetune_data.jsonl` in `main.py:82-84`.

---

### Challenge 7: Streaming Responses (`main.py:94-165`)

```python
for output in app.stream(initial_state, config=config, stream_mode="updates"):
    for node_name, node_value in output.items():
        print(f"✓ Node completed: {node_name}")
        if "final_response" in node_value:
            print(node_value['final_response'])
```

**Command:** `uv run main.py streaming "your query"`

---

### Challenge 8: Multi-language (`agents/nodes.py:384-398`)

```python
prompt = f"""Format the analysis for the user.

Target Language: {language} (Translate your response to this language if it's not English)
...
"""
```

**Detection:** Handled by LLM in `query_processing_node` (lines 49-50).

---

## Running the Agent

### Commands

```bash
# Run test cases
uv run main.py test

# Interactive mode (with persistent memory)
uv run main.py interactive

# Streaming mode demo
uv run main.py streaming

# Bonus features demo
uv run main.py bonus

# Single query
uv run main.py query "Tell me about Javan rhinos"

# Standalone bonus demo
uv run run_bonus.py
```

### Environment Setup

```bash
# Install dependencies
uv sync

# Set Groq API key
export GROQ_API_KEY="your-key-here"
```

---

## Testing

### Test Structure

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_nodes.py` | 13 | Node unit tests |
| `tests/test_workflow.py` | 7 | Integration tests |
| `tests/test_bonus.py` | 12 | Bonus feature tests |

### Running Tests

```bash
# All tests
uv run pytest

# With verbose output
uv run pytest -v

# Specific test file
uv run pytest tests/test_nodes.py

# Coverage report
uv run pytest --cov=agents --cov=bonus_features
```

### Key Test Cases

**Test Queries:**
- "Tell me about the Javan Rhino" → species_info
- "What are the main threats to African Elephants?" → threat_analysis
- "Which animal groups receive the least funding?" → funding_analysis
- "Compare funding for Rhinos vs Pandas" → comparative_analysis
- "How far do wildebeest migrate?" → migration_lookup

---

## Key Concepts

### LangGraph Concepts

| Concept | Implementation |
|---------|----------------|
| **StateGraph** | `workflow = StateGraph(ConservationAgentState)` |
| **TypedDict State** | `ConservationAgentState` with field annotations |
| **Nodes** | Functions that take state and return partial state updates |
| **Edges** | `add_edge(from_node, to_node)` for linear flow |
| **Conditional Edges** | `add_conditional_edges(node, route_fn, {condition: next_node})` |
| **Checkpointing** | `MemorySaver()` for state persistence |
| **Streaming** | `app.stream(state, stream_mode="updates")` |

### LangChain Concepts

| Concept | Implementation |
|---------|----------------|
| **ChatPromptTemplate** | `prompt | llm` chain |
| **LLM Integration** | `ChatGroq(model="llama3-70b-8192")` |
| **Document** | LangChain Document with metadata |
| **VectorStore** | ChromaDB with HuggingFace embeddings |

### Design Patterns

**1. Tool Registry Pattern:**
```python
TOOL_REGISTRY = {"tool_name": tool_function}
tool_func = TOOL_REGISTRY[tool_name]
results = tool_func(args)
```

**2. Fuzzy Matching:**
```python
def fuzzy_match_species(name: str) -> str | None:
    for key in data.keys():
        if name.lower() in key or key in name.lower():
            return key
    return None
```

**3. Error Fallback:**
```python
try:
    # RAG vector search
except:
    # Fall back to full document load
```

**4. Partial State Updates:**
```python
# Nodes only return fields they modify
return {"intent": "species_info", "entities": ["Javan Rhino"]}
# LangGraph merges with existing state
```

---

## Quick Reference

### File Locations

| What | Where |
|------|-------|
| State definition | `agents/state.py:7-46` |
| Tool implementations | `agents/tools.py:13-394` |
| Node functions | `agents/nodes.py:26-424` |
| Graph definition | `agents/graph.py:79-144` |
| Routing logic | `agents/graph.py:23-76` |
| Bonus features | `bonus_features.py` |
| CLI interface | `main.py:281-326` |

### Important Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `create_conservation_agent_graph()` | `agents/graph.py:79` | Build LangGraph |
| `route_to_tool()` | `agents/graph.py:23` | Tool selection routing |
| `check_iteration_limit()` | `agents/graph.py:53` | Loop prevention |
| `query_processing_node()` | `agents/nodes.py:26` | Intent extraction |
| `run_agent()` | `main.py:36` | Main agent entry |
| `run_agent_streaming()` | `main.py:94` | Streaming mode |
| `ConservationRAG.query()` | `bonus_features.py:197` | Vector search |
| `predict_and_visualize_population()` | `bonus_features.py:57` | ML + charts |

---

## Revision Checklist

When reviewing the codebase, verify:

- [ ] State has all required fields (`query`, `intent`, `entities`, etc.)
- [ ] All 6 nodes are implemented with correct signatures
- [ ] Conditional routing handles all edge cases
- [ ] Tools return consistent dictionary formats
- [ ] Error handling uses try/except with fallbacks
- [ ] MemorySaver is enabled (checkpointer)
- [ ] Tests provide thread_id in config
- [ ] Bonus features are optional (try/except imports)
- [ ] JSON keys in `species_data.json` are quoted
- [ ] RAG initializes only if report file exists

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `Checkpointer requires thread_id` | Add `config = {"configurable": {"thread_id": "test_x"}}` |
| `JSONDecodeError in species_data.json` | Ensure keys are quoted: `{"2010": 60}` not `{2010: 60}` |
| Bonus tests skipped | Install full dependencies: `uv sync` |
| RAG not working | Check `data/animal_conservation_report.md` exists |
| Groq API error | Set `GROQ_API_KEY` environment variable |

---

**Last Updated:** 2025-01-27
**LangGraph Version:** 0.2.x
**LLM:** Groq Llama-3-70b-8192
