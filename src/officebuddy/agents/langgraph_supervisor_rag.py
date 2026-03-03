"""
LangGraph Supervisor RAG System using Official Patterns

This implementation uses the official LangGraph supervisor utilities:
- create_react_agent: Creates tool-based agents
- Tool definitions for specialized capabilities
- Proper supervisor pattern from langgraph ecosystem

Reference: https://github.com/langchain-ai/langgraph-supervisor-py
"""

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal

from dotenv import load_dotenv
import os

# ============================================================================
# Environment Setup
# ============================================================================

load_dotenv()
os.environ["USER_AGENT"] = "LangGraphSupervisorRAG/1.0"


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """Shared state for all agents in the system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# ============================================================================
# Vector Store Initialization
# ============================================================================

def initialize_vector_store():
    """Initialize vector store with blog post documents."""
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # Load and split documents
    docs = []
    for url in urls:
        docs.extend(WebBaseLoader(url).load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )

    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Initialize retriever globally
RETRIEVER = initialize_vector_store()


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve relevant documents from the knowledge base.

    Use this tool to search through Lilian Weng's blog posts about:
    - Reward hacking in reinforcement learning
    - Hallucination in AI systems
    - Diffusion video generation
    - Other AI/ML topics

    Args:
        query: The search query to find relevant documents

    Returns:
        str: Retrieved document content
    """
    docs = RETRIEVER.invoke(query)

    result = f"Retrieved {len(docs)} documents:\n\n"
    for i, doc in enumerate(docs, 1):
        result += f"Document {i}:\n{doc.page_content}\n\n"

    return result


@tool
def grade_documents(question: str, documents: str) -> str:
    """
    Grade whether retrieved documents are relevant to the question.

    Use this tool to evaluate if the retrieved documents contain
    useful information for answering the user's question.

    Args:
        question: The user's question
        documents: The retrieved documents to evaluate

    Returns:
        str: Grading result with reasoning
    """
    # Use LLM to grade
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""You are a document grading expert. Evaluate if these documents are relevant to the question.

Question: {question}

Documents:
{documents[:2000]}

Respond with either:
RELEVANT: [brief reasoning]
or
NOT_RELEVANT: [brief reasoning]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def rewrite_query(original_query: str) -> str:
    """
    Rewrite a query to improve document retrieval.

    Use this tool when initial retrieval doesn't find relevant documents.
    It reformulates the query to be more effective for search.

    Args:
        original_query: The original user query that needs improvement

    Returns:
        str: Improved query for better retrieval
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = f"""You are a query optimization expert. Rewrite this query to improve document retrieval.

Original query: {original_query}

Provide an improved version that:
- Expands key concepts
- Adds relevant keywords
- Maintains original intent
- Is optimized for semantic search

Return only the rewritten query."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def generate_answer(question: str, context: str) -> str:
    """
    Generate a final answer based on the question and context.

    Use this tool to synthesize information from retrieved documents
    into a clear, accurate answer.

    Args:
        question: The user's question
        context: Retrieved document context

    Returns:
        str: Generated answer
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = f"""You are an expert at synthesizing information. Generate a clear answer based on the context.

Question: {question}

Context:
{context}

Guidelines:
- Base answer strictly on the provided context
- Be concise (3-5 sentences)
- Cite specific details when relevant
- Acknowledge if context is insufficient

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ============================================================================
# Agent Creation
# ============================================================================

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Create specialized agents using create_react_agent

researcher_agent = create_react_agent(
    model=model,
    tools=[retrieve_documents, rewrite_query],
    name="researcher",
    prompt="""You are a research specialist focused on document retrieval.

Your responsibilities:
- Use retrieve_documents to search the knowledge base
- If initial results aren't good, use rewrite_query to reformulate
- Provide comprehensive document context
- Focus on accuracy and relevance

Always retrieve documents before answering questions about specific topics."""
)

grader_agent = create_react_agent(
    model=model,
    tools=[grade_documents],
    name="grader",
    prompt="""You are a document quality specialist.

Your responsibilities:
- Use grade_documents to evaluate retrieved content
- Provide clear relevance assessments
- Consider both direct and contextual relevance
- Be thorough but fair in your evaluation

Always grade documents before generating answers."""
)

writer_agent = create_react_agent(
    model=model,
    tools=[generate_answer],
    name="writer",
    prompt="""You are an expert writer specializing in clear, accurate answers.

Your responsibilities:
- Use generate_answer to create final responses
- Synthesize information from context
- Maintain accuracy and clarity
- Cite sources when relevant

Only generate answers when you have relevant context."""
)


# ============================================================================
# Supervisor Creation
# ============================================================================

# Define team members
members = ["researcher", "grader", "writer"]

# Routing model
class RouteResponse(BaseModel):
    """Response from supervisor for routing."""
    next: Literal["researcher", "grader", "writer", "FINISH"]


def create_supervisor_node(llm: ChatOpenAI, members: list[str]):
    """
    Create a supervisor node that routes to appropriate agents.

    Args:
        llm: Language model for decision making
        members: List of agent names to route to

    Returns:
        Callable supervisor function
    """
    options = members + ["FINISH"]

    supervisor_prompt = f"""You are a supervisor managing a RAG system with specialized agents.

Your team:
- researcher: Retrieves documents and reformulates queries (tools: retrieve_documents, rewrite_query)
- grader: Evaluates document relevance (tools: grade_documents)
- writer: Generates final answers (tools: generate_answer)

Workflow:
1. Start with researcher to get documents
2. Use grader to check relevance
3. If relevant -> writer for answer
4. If not relevant -> researcher to try again
5. Once answer is generated -> FINISH

For simple greetings, respond directly and FINISH.

Available agents: {options}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", supervisor_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the above conversation, who should act next? Or should we FINISH?"),
    ])

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    def supervisor_node(state: AgentState):
        result = supervisor_chain.invoke(state)
        print(f"🎯 Supervisor routes to: {result.next}")
        return {"next": result.next}

    return supervisor_node


# Create supervisor
supervisor = create_supervisor_node(model, members)


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph():
    """
    Create the supervisor-managed multi-agent graph.

    Returns:
        CompiledGraph: The compiled workflow
    """
    workflow = StateGraph(AgentState)

    # Add supervisor and agent nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("grader", grader_agent)
    workflow.add_node("writer", writer_agent)

    # Define routing
    def route(state: AgentState) -> str:
        """Route based on supervisor's decision."""
        return state.get("next", "supervisor")

    # Add edges
    workflow.add_edge(START, "supervisor")

    # Supervisor routes to agents or finishes
    workflow.add_conditional_edges(
        "supervisor",
        route,
        {
            "researcher": "researcher",
            "grader": "grader",
            "writer": "writer",
            "FINISH": END
        }
    )

    # All agents return to supervisor
    for member in members:
        workflow.add_edge(member, "supervisor")

    return workflow.compile()


# ============================================================================
# Module Export
# ============================================================================

graph = create_graph()
