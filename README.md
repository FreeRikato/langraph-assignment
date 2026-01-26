# LangGraph Conservation Agent

An intelligent multi-step AI agent built with LangGraph that processes, analyzes, and provides insights from the Global Animal Conservation Status Report 2024.

## Overview

This conservation agent demonstrates agentic workflows using LangGraph's node and edge architecture. It can answer questions about endangered species, threats, funding patterns, migration routes, and recovery predictions.

## Features

- **6 Analysis Tools**: Species info, threat analysis, funding analysis, migration lookup, comparative analysis, recovery prediction
- **LangGraph Workflow**: Multi-step reasoning with conditional routing
- **Groq Integration**: Fast LLM inference using Llama 3 70B
- **Fuzzy Matching**: Handles partial species name matches
- **Error Handling**: Graceful degradation when data is missing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Conservation Agent                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   User Query   │────▶│ Query Process  │────▶│ Data Retrieval │
└────────────────┘     └────────────────┘     └────────────────┘
                               │                       │
                               ▼                       ▼
                        Extract Intent          Load Report
                        Extract Entities

                                                      │
                                                      ▼
                                              ┌────────────────┐
                                              │ Tool Selection │
                                              └────────────────┘
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    │                                   │
                                    ▼                                   │
                            ┌────────────────┐                          │
                            │ Tool Execution │◄─────────────────────────┘
                            └────────────────┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │    Analysis    │◄───────┐
                            └────────────────┘        │
                                    │                 (Loop if
                                    ▼                  needed)
                            ┌────────────────┐
                            │  Response Gen  │
                            └────────────────┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │ Final Response │
                            └────────────────┘
```

## Project Structure

```
missoula/
├── main.py                 # Entry point, test runner, interactive mode
├── config.py               # Groq LLM configuration
├── data_loader.py          # Data loading and fuzzy matching utilities
├── agents/
│   ├── __init__.py
│   ├── state.py            # ConservationAgentState TypedDict
│   ├── tools.py            # 6 analysis tools with TOOL_REGISTRY
│   ├── nodes.py            # 6 LangGraph node functions
│   └── graph.py            # StateGraph with edges and routing
├── data/
│   ├── species_data.json   # Species database (10 species)
│   └── animal_conservation_report.md  # RAG document
├── tests/
│   ├── __init__.py
│   ├── test_nodes.py       # Unit tests for nodes
│   └── test_workflow.py    # Integration tests
├── pyproject.toml
├── .env
└── README.md
```

## Setup

### Prerequisites

1. **Python 3.12+** (managed by `uv`)
2. **Groq API Key** - Get one at [console.groq.com](https://console.groq.com)
3. **uv** - Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd missoula
   ```

2. Configure your Groq API key:
   ```bash
   # Create .env file with your key
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

3. Install dependencies (already done if using `uv`):
   ```bash
   uv sync
   ```

## Usage

### Run Test Cases

```bash
uv run main.py test
```

### Interactive Mode

```bash
uv run main.py interactive
```

### Single Query

```bash
uv run main.py query "Tell me about the Javan Rhino"
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agents --cov=main
```

## Agent Capabilities

### Supported Query Types

| Intent | Example Query | Tool Used |
|--------|---------------|-----------|
| Species Info | "Tell me about the Javan Rhino" | `retrieve_species_info` |
| Threat Analysis | "What are the threats to African Elephants?" | `analyze_threats` |
| Funding Analysis | "Which groups receive the least funding?" | `analyze_funding` |
| Migration Lookup | "How far do wildebeest migrate?" | `get_migration_data` |
| Comparative | "Compare rhinos and pandas" | `compare_conservation_metrics` |
| Recovery Prediction | "Can the vaquita recover?" | `predict_recovery_success` |

### Available Species

- Javan Rhino (76 individuals)
- Sumatran Rhino (34 individuals)
- Black Rhino (6,195 individuals)
- African Elephant (415,000 individuals)
- Giant Panda (1,864 individuals)
- Wildebeest (1,500,000 individuals)
- Humpback Whale (84,000 individuals)
- Arabian Oryx (1,200 individuals)
- Snow Leopard (6,390 individuals)
- Vaquita (10 individuals)

## State Management

The agent uses a `ConservationAgentState` TypedDict with the following fields:

```python
class ConservationAgentState(TypedDict):
    query: str                    # User's question
    conversation_history: List    # Chat history
    intent: str                   # Classified intent
    entities: List[str]           # Extracted entities
    retrieved_data: Dict          # Report content
    current_tool: str             # Selected tool
    tool_results: Dict            # Tool output
    analysis_result: str          # LLM analysis
    final_response: str           # Formatted answer
    iteration_count: int          # Loop prevention
    max_iterations: int           # Max iterations (default: 3)
```

## Development

### Adding a New Tool

1. Add the tool function to `agents/tools.py`
2. Register it in `TOOL_REGISTRY`
3. Update intent mapping in `agents/nodes.py`
4. Add intent to the prompt in `query_processing_node`

### Testing

```bash
# Unit tests only
uv run pytest tests/test_nodes.py -v

# Integration tests only
uv run pytest tests/test_workflow.py -v

# Specific test
uv run pytest tests/test_workflow.py::TestWorkflowIntegration::test_species_info_workflow -v
```

## Deliverables Checklist

- [x] Complete LangGraph agent implementation
- [x] All 6 tools implemented and tested
- [x] Proper error handling throughout
- [x] Configuration management (env variables, config.py)
- [x] Clean, documented code with type hints
- [x] README with setup instructions
- [x] Architecture diagram
- [x] Tool documentation
- [x] Test suite documentation
- [x] Unit tests for nodes
- [x] Integration tests for workflows

## Technology Stack

- **LangGraph**: Agentic workflow orchestration
- **LangChain**: LLM integration framework
- **Groq**: High-speed LLM inference (Llama 3 70B)
- **uv**: Modern Python package management
- **Pydantic**: Data validation
- **pytest**: Testing framework

## License

MIT
