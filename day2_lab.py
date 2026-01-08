from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import boto3
import os

# Load variables from .env into environment
load_dotenv()

# Define the AWS CLI Profile to use through env permission
os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE")

# Create a Bedrock Client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Choose a Foundation Model to use from AWS Bedrock
modelID = "us.amazon.nova-lite-v1:0"

# Defining a Model to use in LangChain
llm = ChatBedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={
        "max_tokens_to_sample": 2000,
        "temperature": 0.7
    }
)

# ===========================================
# PART 1: SIMPLE SEQUENTIAL CHAIN
# ===========================================

def build_simple_sequential_chain():
    """Build a two-step chain: generate ideas, then evaluate them"""
    
    idea_prompt = ChatPromptTemplate.from_template(
    "Generate 3 creative app ideas for: {topic}. List them numbered 1-3."
    )
    idea_chain = idea_prompt | llm
    

    eval_prompt = ChatPromptTemplate.from_template(
    "Evaluate these app ideas and pick the best one. Explain why in 2-3 sentences:\n\n{ideas}"
    )
    eval_chain = eval_prompt | llm

    idea_eval_chain = RunnableSequence(idea_chain, eval_chain)
    
    return idea_eval_chain

def test_simple_chain():
    """Test the simple sequential chain"""
    print("\n" + "=" * 50)
    print("🔗 TESTING SIMPLE SEQUENTIAL CHAIN")
    print("=" * 50)
    
    chain = build_simple_sequential_chain()
    
    topics = ["fitness tracking", "language learning"]
    
    for topic in topics:
        print(f"\n📱 Topic: {topic}")
        print("-" * 40)
        result = chain.invoke(topic)
        print(f"🏆 Final Result: {result}")

# ===========================================
# PART 2: ADVANCED SEQUENTIAL CHAIN
# ===========================================

def build_research_chain():
    """Build a three-step research chain with named inputs/outputs"""
    
    # Chain 1: Research the topic
    research_prompt = ChatPromptTemplate.from_template(
        "Research {topic} and provide 3 key facts. Be concise."
    )
    research_chain = research_prompt | llm | StrOutputParser()
    
    
    # Chain 2: Create an outline using the research
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a 3-point outline for an article about {topic} using this research:\n\n{research_data}"
    )
    outline_chain = outline_prompt | llm | StrOutputParser()
    
    # Chain 3: Write a summary using the outline
    summary_prompt = ChatPromptTemplate.from_template(
        "Write a 2-paragraph summary about {topic} following this outline:\n\n{outline}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Advanced sequential chain with named state/outputs
    full_chain = (
        RunnablePassthrough()
        .assign(research_data=research_chain)
        .assign(outline=outline_chain)
        .assign(summary=summary_chain)
    )
    
    return full_chain

def test_research_chain():
    """Test the advanced research chain"""
    print("\n" + "=" * 50)
    print("📚 TESTING ADVANCED RESEARCH CHAIN")
    print("=" * 50)

    chain = build_research_chain()
    
    result = chain.invoke({"topic": "renewable energy"})

    print("\n📊 RESEARCH DATA:")
    print(result["research_data"])
    print("\n📝 OUTLINE:")
    print(result["outline"])
    print("\n📄 SUMMARY:")
    print(result["summary"])

# ===========================================
# PART 3: CONVERSATION MEMORY
# ===========================================

# Global memory store (in production, use a database)
memory_store = {}

def get_session_history(session_id: str):
    """Get or create memory for a session"""
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

def build_memory_chatbot():
    """Build a chatbot that remembers conversations"""
    
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

def test_memory_chatbot():
    """Test the memory-enabled chatbot"""
    print("\n" + "=" * 50)
    print("🧠 TESTING MEMORY CHATBOT")
    print("=" * 50)
    
    chatbot = build_memory_chatbot()
    
    # Create a session config
    config = {"configurable": {"session_id": "user_alice"}}
    
    # Conversation with memory
    messages = [
        "Hi! My name is Alice and I'm a software engineer.",
        "I work at a startup called TechCorp.",
        "Can you summarize what I've told you so far?"
    ]
    
    for msg in messages:
        print(f"\n👤 User: {msg}")
        response = chatbot.invoke({"input": msg}, config=config)
        print(f"🤖 Bot: {response.content}")

def test_multiple_sessions():
    """Test that different users have separate memories"""
    print("\n" + "=" * 50)
    print("👥 TESTING MULTIPLE SESSIONS")
    print("=" * 50)
    
    chatbot = build_memory_chatbot()
    
    # Alice's session
    alice_config = {"configurable": {"session_id": "alice"}}
    
    # Bob's session
    bob_config = {"configurable": {"session_id": "bob"}}
    
    # Alice introduces herself
    print("\n--- Alice's Conversation ---")
    response = chatbot.invoke(
        {"input": "Hi, I'm Alice. I love Python programming."},
        config=alice_config
    )
    print(f"🤖 To Alice: {response.content}")
    
    # Bob introduces himself
    print("\n--- Bob's Conversation ---")
    response = chatbot.invoke(
        {"input": "Hey, I'm Bob. I'm into data science."},
        config=bob_config
    )
    print(f"🤖 To Bob: {response.content}")
    
    # Test Alice's memory
    print("\n--- Testing Alice's Memory ---")
    response = chatbot.invoke(
        {"input": "What do I like?"},
        config=alice_config
    )
    print(f"🤖 To Alice: {response.content}")
    
    # Test Bob's memory
    print("\n--- Testing Bob's Memory ---")
    response = chatbot.invoke(
        {"input": "What am I into?"},
        config=bob_config
    )
    print(f"🤖 To Bob: {response.content}")

# ===========================================
# PART 4: CUSTOM TOOLS
# ===========================================

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

def test_tools():
    """Test tools directly (modern API)"""
    print("\n" + "=" * 50)
    print("🔧 TESTING TOOLS DIRECTLY")
    print("=" * 50)
    
    print("\n📊 Calculator:")
    print(calculator.invoke({"expression": "25 * 4 + 10"}))
    
    print("\n🕐 Current Time:")
    print(get_current_time.invoke({"format_type": "long"}))
    
    print("\n📝 Word Counter:")
    print(word_counter.invoke({"text": "This is a sample sentence to count"}))

# ===========================================
# PART 5: TOOLS + CHAINS INTEGRATION
# ===========================================

from langchain_core.prompts import PromptTemplate

def build_tool_chain():
    """Build a chain that can use tools (modern API)"""
    
    # Modern tools: use the decorated functions directly
    tools = [calculator, get_current_time, word_counter]
    
    # Generate tool descriptions from docstrings
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
    # Chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(
            f"""You have access to these tools:
        {tool_descriptions}

        If you need to use a tool, respond ONLY with:
        TOOL: [tool_name]
        INPUT: [input_for_tool]

        If you don't need to use a tool, answer directly.

        Question: {{question}}"""
        )
    ])
    
    chain = chat_prompt | llm
    
    # Return the chain and a name->tool mapping
    return chain, {t.name: t for t in tools}


def process_with_tools(question: str) -> str:
    """Process a question, using tools if needed (modern API)"""
    
    chain, tools = build_tool_chain()
    
    # Modern API: invoke instead of run
    response = chain.invoke({"question": question})
    text = response.content  # <-- only grab the text
    
    # Check if AI wants to use a tool
    if "TOOL:" in text and "INPUT:" in text:
        lines = text.strip().split("\n")
        tool_name = None
        tool_input = None
        
        for line in lines:
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            if line.startswith("INPUT:"):
                tool_input = line.replace("INPUT:", "").strip()
        
        if tool_name and tool_name in tools:
            # Modern API: invoke the StructuredTool
            tool_obj = tools[tool_name]
            # get the first argument name dynamically
            arg_name = list(tool_obj.args_schema.model_json_schema()['properties'].keys())[0]
            tool_result = tool_obj.invoke({arg_name: tool_input})
            return f"🔧 Used {tool_name}: {tool_result}"  # grab only content
    
    return text

def test_tool_chain():
    """Test the tool-integrated chain"""
    print("\n" + "=" * 50)
    print("🔗 TESTING TOOL CHAIN")
    print("=" * 50)
    
    questions = [
        "What is 15% of 250?",
        "What time is it right now?",
        "How many words are in: The quick brown fox jumps over the lazy dog",
        "What is the capital of France?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        result = process_with_tools(q)
        print(f"💬 Answer: {result}")

# ===========================================
# PART 6: COMPLETE ASSISTANT (Memory + Tools)
# ===========================================

def build_complete_assistant():
    """Build an assistant with both memory and tool awareness"""
    
    tools = [calculator, get_current_time, word_counter]
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"""You are a helpful assistant with memory and tools.

            Available tools:
            {tool_descriptions}

            To use a tool, respond with:
            TOOL: [tool_name]
            INPUT: [input]

            Otherwise, answer naturally. Remember what the user tells you."""
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Combine the prompt with LLM
    chain = prompt | llm
    
    # Wrap with memory-aware runnable
    assistant = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Return both assistant and tool mapping
    return assistant, {t.name: t for t in tools}


def chat_with_assistant(message: str, session_id: str = "default") -> str:
    """Chat with the complete assistant (memory + tools)"""
    
    assistant, tools = build_complete_assistant()
    config = {"configurable": {"session_id": session_id}}
    
    # Invoke the assistant
    response = assistant.invoke({"input": message}, config=config)
    text = response if isinstance(response, str) else getattr(response, "content", str(response))
    
    # Check if the AI wants to use a tool
    if "TOOL:" in text and "INPUT:" in text:
        lines = text.strip().split("\n")
        tool_name = None
        tool_input = None
        
        for line in lines:
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            if line.startswith("INPUT:"):
                tool_input = line.replace("INPUT:", "").strip()
        
        if tool_name and tool_name in tools:
            tool_obj = tools[tool_name]
            
            # Dynamically get first argument name (modern Pydantic v2)
            arg_name = list(tool_obj.args_schema.model_json_schema()['properties'].keys())[0]
            
            # Invoke the tool properly
            tool_result = tool_obj.invoke({arg_name: tool_input})
            
            # Ensure result is a string
            tool_text = tool_result if isinstance(tool_result, str) else getattr(tool_result, "content", str(tool_result))
            
            # Ask assistant to provide a natural response using the tool result
            followup = assistant.invoke(
                {"input": f"The {tool_name} returned: {tool_text}. Please give me a natural response."},
                config=config
            )
            return followup if isinstance(followup, str) else getattr(followup, "content", str(followup))
    
    return text

def test_complete_assistant():
    """Test the complete assistant with memory and tools"""
    print("\n" + "=" * 50)
    print("🤖 TESTING COMPLETE ASSISTANT")
    print("=" * 50)
    
    session = "test_user"
    
    conversation = [
        "Hi! I'm planning a party and my budget is $500.",
        "What time is it?",
        "If I spend 30% of my budget on food, how much is that?",
        "What was my total budget again?"
    ]
    
    for msg in conversation:
        print(f"\n👤 User: {msg}")
        response = chat_with_assistant(msg, session)
        print(f"🤖 Assistant: {response}")

if __name__ == "__main__":
    print("🚀 LangChain Day 2 Lab: Chains, Memory, and Tools")
    print("=" * 60)
    
    # Run all tests
    test_simple_chain()
    test_research_chain()
    test_memory_chatbot()
    test_multiple_sessions()
    test_tools()
    test_tool_chain()
    test_complete_assistant()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")