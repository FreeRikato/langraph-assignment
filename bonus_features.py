"""Bonus features for the conservation agent.

Implements 8 bonus challenges:
1. Multi-modal Analysis (Web Search)
3. Predictive Analytics (ML)
4. Visualization (matplotlib)
5. RAG Integration (ChromaDB)
6. Fine-tuning Prep (JSONL generation)
"""

import io
import base64
import json
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from duckduckgo_search import DDGS

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =============================================================================
# CHALLENGE 1: MULTI-MODAL WEB SEARCH
# =============================================================================

def web_search_tool(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Find real-time conservation news and data using DuckDuckGo search.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of search results with title, url, and body.
    """
    print(f"🔍 Searching web for: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return [{"error": str(e), "query": query}]


# =============================================================================
# CHALLENGE 3 & 4: PREDICTIVE ANALYTICS & VISUALIZATION
# =============================================================================

def predict_and_visualize_population(
    species_name: str,
    historical_data: Dict[int, int]
) -> Dict[str, Any]:
    """
    Predict future population using Linear Regression and generate a chart.

    Args:
        species_name: Name of the species.
        historical_data: Dictionary of year -> population count.

    Returns:
        Dictionary with forecast data, base64-encoded chart, and trend.
    """
    if not historical_data or len(historical_data) < 2:
        return {
            "error": "Insufficient historical data for prediction",
            "forecast": {},
            "chart_base64": "",
            "trend": "Unknown"
        }

    # Prepare data - convert string keys to integers (JSON keys are always strings)
    try:
        years = np.array([int(k) for k in historical_data.keys()]).reshape(-1, 1)
        counts = np.array([float(v) for v in historical_data.values()])
    except (ValueError, TypeError) as e:
        return {
            "error": f"Invalid data format for prediction: {e}",
            "forecast": {},
            "chart_base64": "",
            "trend": "Unknown"
        }

    # Train ML model (Linear Regression)
    model = LinearRegression()
    model.fit(years, counts)

    # Predict next 5 years
    last_year = years.flatten()[-1]
    future_years = np.array([last_year + 1, last_year + 2, last_year + 3, last_year + 4, last_year + 5]).reshape(-1, 1)
    predictions = model.predict(future_years)

    # Determine trend
    trend = "Declining" if model.coef_[0] < 0 else "Increasing" if model.coef_[0] > 0 else "Stable"

    # Generate visualization
    plt.figure(figsize=(10, 6))

    # Historical data points
    plt.scatter(years, counts, color='steelblue', s=100, label='Historical Data', zorder=5)
    plt.plot(years, model.predict(years), color='steelblue', alpha=0.5, linestyle='--')

    # Predictions
    plt.plot(future_years, predictions, color='coral', marker='o', linestyle='-',
             linewidth=2, markersize=8, label='ML Prediction')

    # Formatting
    plt.title(f"Population Forecast: {species_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Population", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "forecast": dict(zip(future_years.flatten().tolist(), [round(float(p), 1) for p in predictions.flatten()])),
        "chart_base64": img_str,
        "trend": trend,
        "r_squared": round(model.score(years, counts), 3),
        "last_known_population": int(counts[-1]),
        "last_known_year": int(last_year)
    }


# =============================================================================
# CHALLENGE 5: RAG INTEGRATION (Vector Store)
# =============================================================================

class ConservationRAG:
    """
    Retrieval-Augmented Generation system for conservation documents.

    Uses ChromaDB with HuggingFace embeddings for semantic search.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory to persist the vector store.
        """
        self.persist_directory = persist_directory
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def ingest_report(self, file_path: str = "data/animal_conservation_report.md") -> None:
        """
        Ingest the conservation report into the vector store.

        Args:
            file_path: Path to the markdown report file.
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Report file not found: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split by markdown headers (##)
        sections = text.split("##")
        docs = []

        for section in sections:
            section = section.strip()
            if section and len(section) > 50:  # Skip very short sections
                # Extract title from first line
                lines = section.split("\n")
                title = lines[0].strip() if lines else "Unknown"

                doc = Document(
                    page_content=section,
                    metadata={"source": "report", "section": title}
                )
                docs.append(doc)

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_fn,
            collection_name="conservation_docs",
            persist_directory=self.persist_directory
        )
        print(f"✅ RAG System Initialized with {len(docs)} document sections")

    def query(self, query: str, k: int = 3) -> str:
        """
        Query the vector store for relevant document sections.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            Concatenated relevant document sections.
        """
        if not self.vector_store:
            return "RAG not initialized. Please call ingest_report() first."

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n\n---\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            print(f"RAG query error: {e}")
            return f"Error querying RAG system: {e}"

    def query_with_scores(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector store and return results with relevance scores.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of results with content and scores.
        """
        if not self.vector_store:
            return [{"error": "RAG not initialized"}]

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
        except Exception as e:
            return [{"error": str(e)}]


# =============================================================================
# CHALLENGE 6: FINE-TUNING DATASET GENERATION
# =============================================================================

def generate_finetuning_dataset(
    conversations: List[Dict[str, str]],
    filename: str = "data/finetune_data.jsonl"
) -> str:
    """
    Convert conversation history into Llama-3/Chat format for fine-tuning.

    Args:
        conversations: List of conversation dictionaries with 'query' and 'response'.
        filename: Output file path for the JSONL dataset.

    Returns:
        Path to the generated file.
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for convo in conversations:
            entry = {
                "messages": [
                    {"role": "system", "content": "You are a conservation science expert providing accurate information about endangered species, threats, and conservation efforts."},
                    {"role": "user", "content": convo.get('query', '')},
                    {"role": "assistant", "content": convo.get('response', '')}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"💾 Fine-tuning dataset saved to {filename} ({len(conversations)} conversations)")
    return filename


def append_to_finetuning_dataset(
    query: str,
    response: str,
    filename: str = "data/finetune_data.jsonl"
) -> None:
    """
    Append a single conversation to the fine-tuning dataset.

    Args:
        query: User query.
        response: Assistant response.
        filename: Dataset file path.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    entry = {
        "messages": [
            {"role": "system", "content": "You are a conservation science expert providing accurate information about endangered species, threats, and conservation efforts."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
    }

    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# =============================================================================
# GLOBAL RAG INSTANCE
# =============================================================================

# Global RAG instance to be initialized at startup
rag_system = ConservationRAG()
