"""Configuration module for the conservation agent."""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


def get_groq_api_key() -> str:
    """Get the Groq API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return api_key


# LLM Configuration
LLM_MODEL = "llama3-70b-8192"  # Llama 3 70B for good speed/intelligence balance
LLM_TEMPERATURE = 0.0  # Deterministic for consistent JSON parsing

# Initialize the ChatGroq model
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=get_groq_api_key(),
)


# Optional: Alternative models for different use cases
def create_llm(
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> ChatGroq:
    """
    Create a new ChatGroq instance with custom settings.

    Args:
        model: Model name to use.
        temperature: Temperature setting.

    Returns:
        Configured ChatGroq instance.
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_groq_api_key(),
    )
