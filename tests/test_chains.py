import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from src.chains import build_simple_sequential_chain, build_research_chain, get_chain


class MockLLM:
    """Mock LLM for testing without AWS credentials"""

    def __call__(self, *args, **kwargs):
        return "Mock response"

    def __or__(self, other):
        """Support pipe operator for LCEL"""
        return other


class TestBuildSimpleSequentialChain:
    """Tests for the simple sequential chain builder"""

    def test_returns_runnable_sequence(self):
        """Verify build_simple_sequential_chain returns a RunnableSequence"""
        mock_llm = MockLLM()
        chain = build_simple_sequential_chain(mock_llm)
        assert isinstance(chain, RunnableSequence)


class TestBuildResearchChain:
    """Tests for the research chain builder"""

    def test_returns_runnable(self):
        """Verify build_research_chain returns a runnable"""
        mock_llm = MockLLM()
        chain = build_research_chain(mock_llm)
        # Should be a runnable (has invoke method)
        assert hasattr(chain, 'invoke')


class TestGetChain:
    """Tests for the chain selector function"""

    def test_get_simple_chain(self):
        """Verify get_chain returns simple chain"""
        mock_llm = MockLLM()
        chain = get_chain("simple", mock_llm)
        assert isinstance(chain, RunnableSequence)

    def test_get_research_chain(self):
        """Verify get_chain returns research chain"""
        mock_llm = MockLLM()
        chain = get_chain("research", mock_llm)
        assert hasattr(chain, 'invoke')

    def test_invalid_chain_raises_error(self):
        """Verify get_chain raises ValueError for unknown chain type"""
        mock_llm = MockLLM()
        with pytest.raises(ValueError) as exc_info:
            get_chain("invalid_chain", mock_llm)

        assert "Unknown chain type" in str(exc_info.value)
        assert "invalid_chain" in str(exc_info.value)
