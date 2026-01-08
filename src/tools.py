"""
Tools module for LangChain application.
Contains custom tools for calculator, time, and word counting.
"""

from langchain.tools import tool
from datetime import datetime


@tool
def calculator(expression: str) -> str:
    """
    Perform a basic math calculation.
    Only supports numbers and + - * / . ( )
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not set(expression).issubset(allowed):
            return "Error: Only basic math operations allowed"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time(format_type: str = "default") -> str:
    """
    Get the current time.
    format_type options: short, long, date, default
    """
    now = datetime.now()
    formats = {
        "short": now.strftime("%H:%M"),
        "long": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "default": now.strftime("%B %d, %Y at %I:%M %p")
    }
    return formats.get(format_type, formats["default"])


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words and characters in a text.
    """
    words = len(text.split())
    chars = len(text)
    return f"Words: {words}, Characters: {chars}"


def get_all_tools() -> list:
    """
    Get a list of all available tools.

    Returns:
        list: List of tool objects
    """
    return [calculator, get_current_time, word_counter]


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all tools.

    Returns:
        str: Formatted tool descriptions
    """
    tools = get_all_tools()
    return "\n".join([f"- {t.name}: {t.description}" for t in tools])


def get_tool_by_name(name: str):
    """
    Get a tool by its name.

    Args:
        name: The tool name

    Returns:
        Tool object or None if not found
    """
    tools = {t.name: t for t in get_all_tools()}
    return tools.get(name)


def invoke_tool(tool_name: str, tool_input: str) -> str:
    """
    Invoke a tool by name with the given input.

    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input string for the tool

    Returns:
        str: Tool result or error message
    """
    tool_obj = get_tool_by_name(tool_name)
    if not tool_obj:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        # Get the first argument name dynamically
        arg_name = list(tool_obj.args_schema.model_json_schema()['properties'].keys())[0]
        result = tool_obj.invoke({arg_name: tool_input})
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        return f"Error invoking tool: {str(e)}"
