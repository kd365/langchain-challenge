import pytest
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import PromptTemplate
from langchain_chatbot_lab import (
    create_assistant_prompt,
    create_summarizer_prompt,
    get_prompt
)


class TestAssistantPrompt:
    """Tests for the assistant prompt template"""

    def test_assistant_prompt_returns_prompt_template(self):
        """Verify create_assistant_prompt returns a PromptTemplate"""
        prompt = create_assistant_prompt()
        assert isinstance(prompt, PromptTemplate)

    def test_assistant_prompt_has_correct_variables(self):
        """Verify assistant prompt has language and freeform_text variables"""
        prompt = create_assistant_prompt()
        assert "language" in prompt.input_variables
        assert "freeform_text" in prompt.input_variables

    def test_assistant_prompt_formats_correctly(self):
        """Verify assistant prompt formats with sample inputs"""
        prompt = create_assistant_prompt()
        formatted = prompt.format(language="English", freeform_text="Hello there")

        assert "English" in formatted
        assert "Hello there" in formatted
        assert "helpful" in formatted.lower()  # Check template text is present


class TestSummarizerPrompt:
    """Tests for the summarizer prompt template"""

    def test_summarizer_prompt_returns_prompt_template(self):
        """Verify create_summarizer_prompt returns a PromptTemplate"""
        prompt = create_summarizer_prompt()
        assert isinstance(prompt, PromptTemplate)

    def test_summarizer_prompt_has_correct_variables(self):
        """Verify summarizer prompt has length and text variables"""
        prompt = create_summarizer_prompt()
        assert "length" in prompt.input_variables
        assert "text" in prompt.input_variables

    def test_summarizer_prompt_formats_correctly(self):
        """Verify summarizer prompt formats with sample inputs"""
        prompt = create_summarizer_prompt()
        formatted = prompt.format(length="brief", text="This is a long article about AI.")

        assert "brief" in formatted
        assert "This is a long article about AI." in formatted
        assert "summarize" in formatted.lower()  # Check template text is present


class TestPromptSelector:
    """Tests for the get_prompt selector function"""

    def test_get_prompt_returns_assistant(self):
        """Verify get_prompt('assistant') returns assistant prompt"""
        prompt = get_prompt("assistant")
        assert isinstance(prompt, PromptTemplate)
        assert "language" in prompt.input_variables

    def test_get_prompt_returns_summarizer(self):
        """Verify get_prompt('summarizer') returns summarizer prompt"""
        prompt = get_prompt("summarizer")
        assert isinstance(prompt, PromptTemplate)
        assert "length" in prompt.input_variables

    def test_get_prompt_invalid_raises_error(self):
        """Verify get_prompt raises ValueError for unknown task"""
        with pytest.raises(ValueError) as exc_info:
            get_prompt("unknown_task")

        assert "Unknown task" in str(exc_info.value)
        assert "unknown_task" in str(exc_info.value)
