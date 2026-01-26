# LangGraph Assignment: Build an Intelligent Animal Conservation Agent

## Assignment Overview

In this assignment, you will build a **multi-step AI agent using LangGraph** that can intelligently process, analyze, and provide insights from the Global Animal Conservation Status Report 2024. The agent will demonstrate agentic workflows, tool integration, and state management using LangGraph’s capabilities.

**Difficulty Level:** Intermediate to Advanced
**Technology Stack:** Python, LangGraph, LangChain, Claude/GPT-4
**Estimated Time:** 4-6 hours
**Team Size:** Individual or pairs

## Learning Objectives

By completing this assignment, you will:

1.  **Design multi-step agentic workflows** using LangGraph node and edge architecture
2.  **Implement tool-calling agents** that interact with external data sources
3.  **Build state management systems** for complex reasoning tasks
4.  **Apply LLM-powered decision logic** to conservation data analysis
5.  **Create reusable agent components** for information retrieval and analysis
6.  **Implement error handling and fallback strategies** in agentic systems
7.  **Develop production-ready patterns** for agent orchestration

---

## Part 1: Foundation Setup (1-1.5 hours)

### 1.1 Project Initialization

Create a new Python project with the following structure:

```text
conservation-agent/
├── main.py
├── agents/
│   ├── init.py
│   └── conservation_agent.py
├── tools.py
├── data/
│   ├── animal_conservation_report.md
│   └── species_data.json
├── config.py
├── requirements.txt
└── README.md
```

### 1.2 Dependencies Installation

Create `requirements.txt` with:

```text
langgraph==0.1.x
langchain==0.1.x
langchain-openai==0.1.x
python-dotenv==1.0.x
requests==2.31.x
pandas==2.1.x
pydantic==2.5.x
```

Install and configure:
*   OpenAI API key (or your preferred LLM)
*   LangGraph for agentic workflows
*   Environment variables in `.env` file

### 1.3 Data Preparation

Parse the provided document and structure data:
*   Extract threatened species information into structured format
*   Parse funding statistics and conservation programs
*   Create JSON representation of migration patterns
*   Build lookup tables for species-specific data

---

## Part 2: Design the Agent Architecture (1.5-2 hours)

### 2.1 Define Agent State

Create a LangGraph State class that tracks:

**ConservationAgentState:**
*   **query:** str (user question)
*   **conversation_history:** list[dict] (message history)
*   **current_tool:** str (active tool name)
*   **retrieved_data:** dict (extracted information)
*   **analysis_result:** str (agent's analysis)
*   **action_taken:** str (last action performed)
*   **next_action:** str (planned next step)

**State Design Considerations:**
*   Use Pydantic models for validation
*   Implement proper typing for state transitions
*   Design immutable state patterns where appropriate
*   Plan for state persistence if needed

### 2.2 Define Agent Nodes

Create the following LangGraph nodes:

**Node 1: Query Processing**
*   **Input:** User question about animal conservation
*   **Processing:** Parse query intent and extract entities
*   **Output:** Structured query parameters
*   **Purpose:** Transform natural language to machine-readable format

**Node 2: Data Retrieval**
*   **Input:** Structured query parameters
*   **Processing:** Search conservation database for relevant information
*   **Output:** Retrieved data documents and metadata
*   **Purpose:** Fetch relevant information from document

**Node 3: Tool Selection**
*   **Input:** Query intent and retrieved data
*   **Processing:** LLM decides which tool(s) to use next
*   **Output:** Selected tool name and parameters
*   **Purpose:** Dynamic tool routing based on query

**Node 4: Tool Execution**
*   **Input:** Tool name and parameters
*   **Processing:** Execute selected analysis or retrieval tool
*   **Output:** Tool results
*   **Purpose:** Perform specific conservation analysis

**Node 5: Analysis & Reasoning**
*   **Input:** Retrieved data and tool results
*   **Processing:** LLM synthesizes information and generates insights
*   **Output:** Structured analysis with reasoning
*   **Purpose:** Generate meaningful conservation insights

**Node 6: Response Generation**
*   **Input:** Analysis results
*   **Processing:** Format findings into user-friendly response
*   **Output:** Final response to user
*   **Purpose:** Present findings clearly

### 2.3 Define Agent Edges

Create conditional routing between nodes:
1.  **Query Processing -> Data Retrieval:** Always transition
2.  **Data Retrieval -> Tool Selection:** Conditional based on data availability
3.  **Tool Selection -> Tool Execution:** Route to appropriate tool
4.  **Tool Execution -> Analysis & Reasoning:** Process tool results
5.  **Analysis & Reasoning -> Response Generation:** With confidence > threshold
6.  **Analysis & Reasoning -> Tool Selection:** Loop if more tools needed (max 3 iterations)
7.  **Response Generation -> END:** Final response ready

---

## Part 3: Implement Agent Tools (1.5-2 hours)

### 3.1 Tool 1: Species Information Retrieval
**Purpose:** Fetch detailed information about specific endangered species

**Function Signature:**
`retrieve_species_info(species_name: str) -> dict`

**Implementation Requirements:**
*   Search species database by common or scientific name
*   Return: population count, threat level, primary threats, conservation status
*   Handle partial name matches with similarity scoring
*   Return top 3 matches if exact match not found

**Example Query:** "Tell me about the Javan Rhino"

### 3.2 Tool 2: Threat Analysis
**Purpose:** Analyze primary threats affecting species or regions

**Function Signature:**
`analyze_threats(target: str, threat_type: str) -> dict`

**Implementation Requirements:**
*   Analyze threats by species, region, or taxonomic group
*   Calculate threat severity scores
*   Identify correlations between threats
*   Provide mitigation recommendations

**Threat Categories:** Habitat loss, poaching, climate change, pollution, invasive species
**Example Query:** "What are the main threats to African elephants?"

### 3.3 Tool 3: Funding Analysis
**Purpose:** Analyze conservation funding patterns and gaps

**Function Signature:**
`analyze_funding(category: str, filters: dict) -> dict`

**Implementation Requirements:**
*   Query funding data by species, region, or taxonomic group
*   Calculate funding gaps and allocations
*   Identify underfunded areas
*   Project future funding needs

**Example Query:** "Which animal groups receive the least funding?"

### 3.4 Tool 4: Migration Pattern Lookup
**Purpose:** Retrieve migration data for species

**Function Signature:**
`get_migration_data(species_name: str) -> dict`

**Implementation Requirements:**
*   Return migration routes, distance, duration
*   Provide population data during migration
*   Identify seasonal patterns
*   Highlight conservation challenges during migration

**Example Query:** "How far do wildebeest migrate annually?"

### 3.5 Tool 5: Comparative Analysis
**Purpose:** Compare conservation metrics across species or regions

**Function Signature:**
`compare_conservation_metrics(entities: list[str], metric: str) -> dict`

**Implementation Requirements:**
*   Compare population trends, funding, threat levels
*   Generate comparison tables with statistics
*   Calculate correlation metrics
*   Provide ranking analysis

**Example Query:** "Compare funding levels for rhino species"

### 3.6 Tool 6: Recovery Success Prediction
**Purpose:** Predict recovery success based on conservation data

**Function Signature:**
`predict_recovery_success(species_name: str, intervention: str) -> dict`

**Implementation Requirements:**
*   Use historical recovery data (Arabian Oryx, Humpback Whale, etc.)
*   Calculate success probability based on intervention type
*   Identify key success factors
*   Estimate timeline to recovery

**Example Query:** "What's the likelihood of black rhino recovery if we increase funding?"

---

## Part 4: Build the LangGraph Workflow (1.5-2 hours)

### 4.1 Graph Definition

```python
from langgraph.graph import StateGraph, END

def create_conservation_agent():
    graph = StateGraph(ConservationAgentState)

    # Add nodes
    graph.add_node("query_processing", query_processing_node)
    graph.add_node("data_retrieval", data_retrieval_node)
    graph.add_node("tool_selection", tool_selection_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("response_generation", response_generation_node)

    # Add edges with routing logic
    graph.add_edge("query_processing", "data_retrieval")
    graph.add_conditional_edges(
        "data_retrieval",
        route_to_tool_selection,
        {
            "tool_selection": "tool_selection",
            "direct_response": "response_generation"
        }
    )
    # ... additional edge definitions

    # Set entry point and compile
    graph.set_entry_point("query_processing")
    return graph.compile()
```

### 4.2 Conditional Routing Implementation

1.  **route_to_tool_selection:** Decide if tool execution needed
2.  **route_tool_selection:** Map query intent to specific tool
3.  **route_loop_or_continue:** Decide loop back or proceed to analysis
4.  **route_to_output:** Determine final response path

### 4.3 Error Handling & Fallbacks

Implement robust error handling:
*   **Node failures:** Retry logic with exponential backoff
*   **Tool execution errors:** Fallback to alternative tools
*   **LLM API failures:** Graceful degradation with cached responses
*   **Invalid queries:** User-friendly error messages and suggestions
*   **Data not found:** Alternative search strategies (fuzzy matching, semantic search)

---

## Part 5: Implement Node Logic (2-2.5 hours)

### 5.1 Query Processing Node
```python
def query_processing_node(state: ConservationAgentState) -> ConservationAgentState:
    """Parse and structure user query"""
    # Extract intent (species_info, threat_analysis, funding_analysis, etc.)
    # Extract entities (species names, regions, taxonomic groups)
    # Determine required tools
    # Set state for next node
```
**Key Responsibilities:**
*   Intent classification (6+ intent types)
*   Named entity recognition for species, regions
*   Multi-intent detection (handle complex queries)
*   Confidence scoring for each detected intent

### 5.2 Data Retrieval Node
```python
def data_retrieval_node(state: ConservationAgentState) -> ConservationAgentState:
    """Retrieve relevant data from conservation database"""
    # Query document/database based on parsed query
    # Return structured data
    # Assess data sufficiency
```
**Key Responsibilities:**
*   Semantic search over conservation document
*   Structured data lookup from JSON files
*   Data validation and enrichment
*   Return ranked results by relevance

### 5.3 Tool Selection Node
```python
def tool_selection_node(state: ConservationAgentState) -> ConservationAgentState:
    """Use LLM to decide which tool(s) to invoke"""
    # Call LLM with query context and available tools
    # Parse tool selection from LLM response
    # Prepare tool parameters
```
**Key Responsibilities:**
*   Present available tools to LLM
*   Parse structured tool selection response
*   Multi-tool sequencing support
*   Parameter validation

### 5.4 Tool Execution Node
```python
def tool_execution_node(state: ConservationAgentState) -> ConservationAgentState:
    """Execute selected tool and capture results"""
    # Route to appropriate tool function
    # Handle tool execution
    # Store results in state
```
**Key Responsibilities:**
*   Dynamic tool invocation
*   Result formatting and validation
*   Error capture and logging
*   Tool-specific result processing

### 5.5 Analysis & Reasoning Node
```python
def analysis_node(state: ConservationAgentState) -> ConservationAgentState:
    """Synthesize information and generate insights"""
    # Use LLM to analyze tool results
    # Generate reasoning chain
    # Decide if more tools needed
```
**Key Responsibilities:**
*   Multi-step reasoning over tool results
*   Insight generation with supporting evidence
*   Decision logic for additional tool calls
*   Confidence assessment

### 5.6 Response Generation Node
```python
def response_generation_node(state: ConservationAgentState) -> ConservationAgentState:
    """Format final response for user"""
    # Structure response with findings
    # Add supporting data/citations
    # Format for readability
```
**Key Responsibilities:**
*   Response structuring (sections, formatting)
*   Citation and source attribution
*   Data visualization recommendations
*   Next question suggestions

---

## Part 6: Testing & Validation (1-1.5 hours)

### 6.1 Unit Tests
Create tests for each node:
```python
def test_query_processing_node():
    # Test various query intents
    # Verify entity extraction
    # Check confidence scoring

def test_tool_execution():
    # Test each tool independently
    # Verify error handling
    # Check result formatting

def test_conditional_routing():
    # Test edge conditions
    # Verify routing logic
    # Check state transitions
```

### 6.2 Integration Tests
Test complete workflows:
1.  **End-to-end query flow:** User query -> agent response
2.  **Multi-tool sequences:** Complex queries requiring multiple tools
3.  **Error scenarios:** Handle missing data, API failures
4.  **Loop prevention:** Ensure agent doesn't enter infinite loops
5.  **State consistency:** Verify state maintained correctly

### 6.3 Test Queries
Validate agent with diverse questions:

**Species Information:**
*   "Tell me about the Javan Rhino"
*   "What's the current population of giant pandas?"
*   "Which rhino species is most endangered?"

**Threat Analysis:**
*   "What are the main threats to African elephants?"
*   "How does climate change affect animal migration?"
*   "Which regions have the most threatened species?"

**Funding Analysis:**
*   "How is conservation funding distributed across species?"
*   "Which animal groups are underfunded?"
*   "What's the total funding gap for endangered species?"

**Migration & Complex Queries:**
*   "Explain the wildebeest migration and its conservation challenges"
*   "Compare funding vs. threat levels for different species"
*   "What interventions would most help Javan rhino recovery?"

---

## Part 7: Production Considerations (Documentation)

### 7.1 Deployment Patterns
*   **API Service:** Deploy agent as FastAPI endpoint
*   **Async Processing:** Handle long-running analyses
*   **Caching:** Cache frequently accessed data and tool results
*   **Monitoring:** Log agent decisions and tool usage patterns
*   **Rate Limiting:** Manage API costs

### 7.2 Scalability & Performance
*   Optimize tool execution for large datasets
*   Implement batch processing for multiple queries
*   Design for concurrent agent instances
*   Monitor token usage for LLM API costs

### 7.3 Documentation
*   Document each tool's parameters and return format
*   Create usage examples for peers
*   Include troubleshooting guide
*   Provide architecture diagrams

---

## Deliverables Checklist

**Code Submission (50%):**
*   [ ] Complete LangGraph agent implementation
*   [ ] All 6+ tools implemented and tested
*   [ ] Proper error handling throughout
*   [ ] Configuration management (env variables, config files)
*   [ ] Clean, documented code with type hints

**Documentation (30%):**
*   [ ] README with setup instructions
*   [ ] Architecture diagram (nodes, edges, data flow)
*   [ ] Tool documentation with examples
*   [ ] Test suite documentation
*   [ ] API documentation if deployed as service

**Testing & Validation (20%):**
*   [ ] Unit tests for each node (minimum 80% coverage)
*   [ ] Integration tests covering main workflows
*   [ ] Test results on diverse query types
*   [ ] Error handling demonstrations
*   [ ] Performance metrics (execution time, token usage)

---

## Evaluation Criteria

**Functionality (40%):**
*   Agent successfully routes queries to appropriate tools
*   Tools return accurate information from document
*   Multi-step reasoning works correctly
*   Error handling and edge cases addressed

**Code Quality (30%):**
*   Clear architecture with separation of concerns
*   Proper use of LangGraph patterns and constructs
*   Comprehensive error handling
*   Well-documented and maintainable code

**Innovation (20%):**
*   Creative tool implementations beyond basic requirements
*   Interesting analysis or insights generated
*   Optimization or performance improvements
*   Enhanced user experience features

**Testing & Documentation (10%):**
*   Comprehensive test coverage
*   Clear API and usage documentation
*   Deployment/running instructions
*   Architecture decisions explained

---

## Bonus Challenges
**Extended Scope (for advanced learners):**

1.  **Multi-modal Analysis:** Integrate with web search to find real-time species updates
2.  **Persistent Learning:** Track conversation history and learn from user feedback
3.  **Predictive Analytics:** Build ML model to predict future threat levels
4.  **Visualization:** Generate charts and migration route visualizations
5.  **RAG Integration:** Implement full Retrieval-Augmented Generation pipeline
6.  **Fine-tuning:** Fine-tune LLM on conservation domain data
7.  **Streaming Responses:** Implement token streaming for real-time output
8.  **Multi-language Support:** Handle queries in multiple languages
