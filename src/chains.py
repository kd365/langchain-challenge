"""
Chain building module for LangChain application.
Contains functions for creating sequential and research chains.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_simple_sequential_chain(llm):
    """
    Build a two-step chain: generate ideas, then evaluate them.

    Args:
        llm: The language model to use

    Returns:
        RunnableSequence: A chain that generates and evaluates app ideas
    """
    # Step 1: Generate ideas
    idea_prompt = ChatPromptTemplate.from_template(
        "Generate 3 creative app ideas for: {topic}. List them numbered 1-3."
    )
    idea_chain = idea_prompt | llm

    # Step 2: Evaluate the ideas
    eval_prompt = ChatPromptTemplate.from_template(
        "Evaluate these app ideas and pick the best one. Explain why in 2-3 sentences:\n\n{ideas}"
    )
    eval_chain = eval_prompt | llm

    # Combine into sequential chain
    idea_eval_chain = RunnableSequence(idea_chain, eval_chain)

    return idea_eval_chain


def build_research_chain(llm):
    """
    Build a three-step research chain with named inputs/outputs.

    Steps:
    1. Research the topic
    2. Create an outline
    3. Write a summary

    Args:
        llm: The language model to use

    Returns:
        Runnable: A chain that produces research_data, outline, and summary
    """
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


def get_chain(chain_type, llm):
    """
    Chain selector function - returns appropriate chain based on type.

    Args:
        chain_type (str): Either "simple" or "research"
        llm: The language model to use

    Returns:
        Runnable: The appropriate chain

    Raises:
        ValueError: If chain_type is not recognized
    """
    chains = {
        "simple": build_simple_sequential_chain,
        "research": build_research_chain
    }

    if chain_type not in chains:
        raise ValueError(f"Unknown chain type: {chain_type}. Available: {list(chains.keys())}")

    return chains[chain_type](llm)
