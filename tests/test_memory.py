import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.chat_history import InMemoryChatMessageHistory
from src.memory import (
    get_session_history,
    clear_session,
    list_sessions,
    memory_store
)


class TestGetSessionHistory:
    """Tests for session history management"""

    def setup_method(self):
        """Clear memory store before each test"""
        memory_store.clear()

    def test_creates_new_session(self):
        """Verify get_session_history creates a new session"""
        history = get_session_history("test_session_1")
        assert isinstance(history, InMemoryChatMessageHistory)

    def test_returns_same_session(self):
        """Verify get_session_history returns same session on repeated calls"""
        history1 = get_session_history("test_session_2")
        history2 = get_session_history("test_session_2")
        assert history1 is history2

    def test_different_sessions_are_independent(self):
        """Verify different session IDs create different histories"""
        history1 = get_session_history("session_a")
        history2 = get_session_history("session_b")
        assert history1 is not history2


class TestClearSession:
    """Tests for clearing sessions"""

    def setup_method(self):
        """Clear memory store before each test"""
        memory_store.clear()

    def test_clear_existing_session(self):
        """Verify clear_session removes an existing session"""
        get_session_history("to_clear")
        assert "to_clear" in memory_store

        result = clear_session("to_clear")
        assert result is True
        assert "to_clear" not in memory_store

    def test_clear_nonexistent_session(self):
        """Verify clear_session returns False for nonexistent session"""
        result = clear_session("does_not_exist")
        assert result is False


class TestListSessions:
    """Tests for listing sessions"""

    def setup_method(self):
        """Clear memory store before each test"""
        memory_store.clear()

    def test_list_empty_sessions(self):
        """Verify list_sessions returns empty list when no sessions"""
        sessions = list_sessions()
        assert sessions == []

    def test_list_multiple_sessions(self):
        """Verify list_sessions returns all session IDs"""
        get_session_history("user1")
        get_session_history("user2")
        get_session_history("user3")

        sessions = list_sessions()
        assert len(sessions) == 3
        assert "user1" in sessions
        assert "user2" in sessions
        assert "user3" in sessions
