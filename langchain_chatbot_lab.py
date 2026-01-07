from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
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
        "temperature": 0.9
    }
)

def create_assistant_prompt():
    """General multilingual assistant prompt"""
    return PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a helpful and friendly chatbot. You are communicating in {language}.\n\n{freeform_text}"
    )

def create_summarizer_prompt():
    """Summarizer prompt with length parameter (brief/detailed)"""
    return PromptTemplate(
        input_variables=["length", "text"],
        template="""You are a text summarizer. Provide a {length} summary of the following text.

If length is "brief": Provide 1-2 sentences capturing the main point.
If length is "detailed": Provide a comprehensive summary with key points and details.

Text to summarize:
{text}

Summary:"""
    )

def get_prompt(task_name):
    """
    Prompt selector function - returns appropriate prompt template based on task name.

    Args:
        task_name (str): Either "assistant" or "summarizer"

    Returns:
        PromptTemplate: The appropriate prompt template

    Raises:
        ValueError: If task_name is not recognized
    """
    prompts = {
        "assistant": create_assistant_prompt,
        "summarizer": create_summarizer_prompt
    }

    if task_name not in prompts:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(prompts.keys())}")

    return prompts[task_name]()

def my_chatbot(language, freeform_text):
    """
    Main chatbot function that processes user input and returns AI response

    Args:
        language (str): The language for the chatbot to respond in
        freeform_text (str): The user's input/question

    Returns:
        str: The AI's response
    """
    # Get our prompt template
    prompt = get_prompt("assistant")

    # Chain using LCEL: prompt → model → output parser
    chain = prompt | llm | StrOutputParser()

    # Invoke chain with our inputs - returns clean string directly
    response = chain.invoke({
        'language': language,
        'freeform_text': freeform_text
    })

    return response

def my_summarizer(length, text):
    """
    Summarizer function that summarizes text

    Args:
        length (str): Either "brief" or "detailed"
        text (str): The text to summarize

    Returns:
        str: The summarized text
    """
    prompt = get_prompt("summarizer")

    # Chain using LCEL: prompt → model → output parser
    chain = prompt | llm | StrOutputParser()

    # Returns clean string directly (no need for .content)
    response = chain.invoke({
        'length': length,
        'text': text
    })
    return response

def test_chatbot():
    """Test function to try out different scenarios"""

    test_cases = [
        ("English", "Which are better dogs, Chihuahuas or Bulldogs?"),
        ("Spanish", "Explícame qué es la inteligencia artificial"),
        ("French", "Raconte-moi une blague"),
        ("English", "Write a haiku about programming"),
    ]

    print("🤖 Testing Multilingual Chatbot")
    print("=" * 40)

    for language, question in test_cases:
        print(f"\n🌍 Language: {language}")
        print(f"❓ Question: {question}")

        try:
            response = my_chatbot(language, question)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print("-" * 40)

def interactive_mode():
    """Interactive session with mode selection"""
    print("🎯 Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 30)

    while True:
        mode = input("\n📋 Select mode (assistant/summarizer): ").strip().lower()
        if mode == 'quit':
            break

        if mode == 'assistant':
            language = input("🌍 Enter language (English/Spanish/French/etc.): ").strip()
            if language.lower() == 'quit':
                break
            user_input = input("💬 Your message: ").strip()
            if user_input.lower() == 'quit':
                break
            try:
                response = my_chatbot(language, user_input)
                print(f"🤖 Bot: {response}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

        elif mode == 'summarizer':
            length = input("📏 Summary length (brief/detailed): ").strip().lower()
            if length == 'quit':
                break
            text = input("📝 Text to summarize: ").strip()
            if text.lower() == 'quit':
                break
            try:
                response = my_summarizer(length, text)
                print(f"📄 Summary: {response}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

        else:
            print("❌ Invalid mode. Choose 'assistant' or 'summarizer'.")

    print("👋 Goodbye!")

if __name__ == "__main__":
    # test_chatbot()  # Comment out for interactive mode
    interactive_mode()

