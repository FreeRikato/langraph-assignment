"""Configuration module for the conservation agent."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def get_zai_api_key() -> str:
    """Get the Z.AI API key from environment."""
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ValueError("ZAI_API_KEY not found in environment variables")
    return api_key


# LLM Configuration
LLM_MODEL = "GLM-4.7"  # GLM-4.7 for coding tasks
LLM_TEMPERATURE = 0.0  # Deterministic for consistent JSON parsing

# Initialize the ChatOpenAI model with Z.AI configuration
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=get_zai_api_key(),
    base_url="https://api.z.ai/api/coding/paas/v4",  # Z.AI Coding API endpoint
)


# Optional: Alternative models for different use cases
def create_llm(
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> ChatOpenAI:
    """
    Create a new ChatOpenAI instance with custom settings.

    Args:
        model: Model name to use (e.g., "GLM-4.7", "GLM-4.5-air").
        temperature: Temperature setting.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=get_zai_api_key(),
        base_url="https://api.z.ai/api/coding/paas/v4",
    )
