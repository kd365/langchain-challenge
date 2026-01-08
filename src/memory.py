"""
Memory module for LangChain application.
Handles conversation history and session management.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Global memory store (in production, use a database)
memory_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Get or create memory for a session.

    Args:
        session_id: Unique identifier for the conversation session

    Returns:
        InMemoryChatMessageHistory: The chat history for this session
    """
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]


def clear_session(session_id: str) -> bool:
    """
    Clear the memory for a specific session.

    Args:
        session_id: The session to clear

    Returns:
        bool: True if session was cleared, False if it didn't exist
    """
    if session_id in memory_store:
        del memory_store[session_id]
        return True
    return False


def list_sessions() -> list:
    """
    List all active session IDs.

    Returns:
        list: List of session IDs
    """
    return list(memory_store.keys())


def build_memory_chatbot(llm):
    """
    Build a chatbot that remembers conversations.

    Args:
        llm: The language model to use

    Returns:
        RunnableWithMessageHistory: A memory-enabled chatbot
    """
    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You remember everything the user tells you."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Create the base chain
    chain = prompt | llm

    # Wrap with memory
    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chatbot


def chat(chatbot, message: str, session_id: str = "default") -> str:
    """
    Send a message to the chatbot and get a response.

    Args:
        chatbot: The memory-enabled chatbot
        message: The user's message
        session_id: The session identifier

    Returns:
        str: The chatbot's response
    """
    config = {"configurable": {"session_id": session_id}}
    response = chatbot.invoke({"input": message}, config=config)

    # Handle both string and AIMessage responses
    if hasattr(response, 'content'):
        return response.content
    return str(response)
