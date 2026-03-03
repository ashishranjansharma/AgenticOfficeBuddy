"""
Official LangGraph Supervisor RAG using create_supervisor

This implementation uses the official create_supervisor utility from langgraph-supervisor.
It's the cleanest and most recommended way to build multi-agent systems.

Reference: https://github.com/langchain-ai/langgraph-supervisor-py
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain.agents import create_agent
from langgraph_supervisor.supervisor import create_supervisor

from dotenv import load_dotenv
import os

# Import all prompts from centralized prompts module
from officebuddy.agents.prompts import (
    RESEARCHER_AGENT_PROMPT,
    GRADER_AGENT_PROMPT,
    WRITER_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    REWRITE_QUERY_TOOL_PROMPT,
    GRADE_DOCUMENTS_TOOL_PROMPT,
    GENERATE_ANSWER_TOOL_PROMPT,
)

# Import persistent vector store
from officebuddy.vector_store import get_retriever


# ============================================================================
# Environment Setup
# ============================================================================

load_dotenv()
os.environ["USER_AGENT"] = "OfficeBuddyRAG/1.0"


# ============================================================================
# Vector Store Initialization
# ============================================================================

# Initialize retriever globally using persistent FAISS vector store
# This will load from cache if available, or build and cache if needed
RETRIEVER = get_retriever(k=4)


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve relevant documents from Lilian Weng's blog posts.

    Search through blog posts about:
    - Reward hacking in reinforcement learning
    - Hallucination in AI systems
    - Diffusion video generation

    Args:
        query: Search query to find relevant documents

    Returns:
        str: Retrieved document content
    """
    docs = RETRIEVER.invoke(query)

    result = f"Retrieved {len(docs)} relevant documents:\n\n"
    for i, doc in enumerate(docs, 1):
        result += f"--- Document {i} ---\n{doc.page_content}\n\n"

    return result


@tool
def rewrite_query(original_query: str) -> str:
    """
    Rewrite a query to improve document retrieval.

    Use when initial retrieval doesn't find relevant documents.

    Args:
        original_query: The query that needs improvement

    Returns:
        str: Improved query for better retrieval
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = REWRITE_QUERY_TOOL_PROMPT.format(original_query=original_query)
    response = llm.invoke([HumanMessage(content=prompt)])

    return f"Rewritten query: {response.content}"


@tool
def grade_documents(question: str, documents: str) -> str:
    """
    Evaluate if retrieved documents are relevant to the question.

    Args:
        question: The user's question
        documents: Retrieved documents to evaluate

    Returns:
        str: Assessment of relevance
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = GRADE_DOCUMENTS_TOOL_PROMPT.format(
        question=question,
        documents=documents[:1500]
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content


@tool
def generate_answer(question: str, context: str) -> str:
    """
    Generate a clear answer based on the question and retrieved context.

    Args:
        question: The user's question
        context: Retrieved document context

    Returns:
        str: Generated answer
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = GENERATE_ANSWER_TOOL_PROMPT.format(
        question=question,
        context=context
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content


# ============================================================================
# Agent Creation using create_react_agent
# ============================================================================

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Researcher Agent - Retrieves and searches documents
researcher_agent = create_agent(
    model=model,
    tools=[retrieve_documents, rewrite_query],
    name="researcher",
    system_prompt=RESEARCHER_AGENT_PROMPT
)

# Grader Agent - Evaluates document quality
grader_agent = create_agent(
    model=model,
    tools=[grade_documents],
    name="grader",
    system_prompt=GRADER_AGENT_PROMPT
)

# Writer Agent - Generates final answers
writer_agent = create_agent(
    model=model,
    tools=[generate_answer],
    name="writer",
    system_prompt=WRITER_AGENT_PROMPT
)


# ============================================================================
# Supervisor Creation using create_supervisor (THE OFFICIAL WAY!)
# ============================================================================

# Create supervisor workflow using the official utility
workflow = create_supervisor(
    agents=[researcher_agent, grader_agent, writer_agent],
    model=model,
    prompt=SUPERVISOR_PROMPT
)


# ============================================================================
# Graph Compilation
# ============================================================================

# Compile the graph
graph = workflow.compile()


# ============================================================================
# Helper Function for Easy Testing
# ============================================================================

def run_query(question: str):
    """
    Helper function to run a query through the RAG system.

    Args:
        question: The user's question

    Returns:
        dict: The final state with messages
    """
    result = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    return result


if __name__ == "__main__":
    # Test the system
    print("🤖 Official LangGraph Supervisor RAG System")
    print("=" * 70)

    test_query = "What does Lilian Weng say about types of reward hacking?"
    print(f"\n📝 Query: {test_query}")
    print("-" * 70)

    result = run_query(test_query)

    print("\n✅ Final Answer:")
    print("-" * 70)
    if result.get("messages"):
        final_message = result["messages"][-1]
        print(final_message.content)
    print("=" * 70)
