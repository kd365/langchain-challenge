import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import (
    calculator,
    get_current_time,
    word_counter,
    get_all_tools,
    get_tool_descriptions,
    get_tool_by_name,
    invoke_tool
)


class TestCalculator:
    """Tests for the calculator tool"""

    def test_basic_addition(self):
        """Verify calculator handles addition"""
        result = calculator.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_basic_multiplication(self):
        """Verify calculator handles multiplication"""
        result = calculator.invoke({"expression": "5 * 3"})
        assert "15" in result

    def test_complex_expression(self):
        """Verify calculator handles complex expressions"""
        result = calculator.invoke({"expression": "(10 + 5) * 2"})
        assert "30" in result

    def test_invalid_characters_rejected(self):
        """Verify calculator rejects invalid characters"""
        result = calculator.invoke({"expression": "import os"})
        assert "Error" in result


class TestGetCurrentTime:
    """Tests for the time tool"""

    def test_default_format(self):
        """Verify get_current_time returns default format"""
        result = get_current_time.invoke({"format_type": "default"})
        assert "at" in result  # Default format includes "at"

    def test_short_format(self):
        """Verify get_current_time returns short format"""
        result = get_current_time.invoke({"format_type": "short"})
        assert ":" in result  # Short format is HH:MM

    def test_date_format(self):
        """Verify get_current_time returns date format"""
        result = get_current_time.invoke({"format_type": "date"})
        assert "-" in result  # Date format is YYYY-MM-DD


class TestWordCounter:
    """Tests for the word counter tool"""

    def test_counts_words(self):
        """Verify word_counter counts words correctly"""
        result = word_counter.invoke({"text": "one two three"})
        assert "Words: 3" in result

    def test_counts_characters(self):
        """Verify word_counter counts characters correctly"""
        result = word_counter.invoke({"text": "hello"})
        assert "Characters: 5" in result

    def test_empty_string(self):
        """Verify word_counter handles empty string"""
        result = word_counter.invoke({"text": ""})
        assert "Words: 0" in result


class TestToolHelpers:
    """Tests for tool helper functions"""

    def test_get_all_tools_returns_list(self):
        """Verify get_all_tools returns a list of tools"""
        tools = get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_get_tool_descriptions_returns_string(self):
        """Verify get_tool_descriptions returns formatted string"""
        descriptions = get_tool_descriptions()
        assert isinstance(descriptions, str)
        assert "calculator" in descriptions
        assert "get_current_time" in descriptions
        assert "word_counter" in descriptions

    def test_get_tool_by_name_found(self):
        """Verify get_tool_by_name returns tool when found"""
        tool = get_tool_by_name("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_tool_by_name_not_found(self):
        """Verify get_tool_by_name returns None when not found"""
        tool = get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_invoke_tool_success(self):
        """Verify invoke_tool works with valid tool"""
        result = invoke_tool("calculator", "1 + 1")
        assert "2" in result

    def test_invoke_tool_unknown(self):
        """Verify invoke_tool handles unknown tool"""
        result = invoke_tool("unknown", "test")
        assert "Error" in result
        assert "Unknown tool" in result
