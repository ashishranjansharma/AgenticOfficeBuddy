"""
Multi-Agent RAG System with LangGraph Supervisor Pattern

This implementation follows the official LangGraph supervisor pattern where:
- A supervisor LLM orchestrates multiple specialized agents
- Each agent is a separate node with specific capabilities
- Agents communicate through a shared state
- The supervisor routes tasks based on capabilities and context

Architecture based on: https://github.com/langchain-ai/langgraph-supervisor-py

Agents:
1. Researcher: Retrieves and searches through documents
2. Grader: Evaluates document quality and relevance
3. Writer: Generates well-crafted answers from context
"""

import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

# ============================================================================
# Environment Setup
# ============================================================================

load_dotenv()
os.environ["USER_AGENT"] = "MultiAgentRAG/1.0"

# ============================================================================
# Agent State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    State shared across all agents in the system.

    The supervisor and all agents can read from and write to this state.
    """
    # The conversation history - agents append their messages here
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The next agent to call (determined by supervisor)
    next: str
    # Documents retrieved from vector store
    retrieved_documents: str
    # Current search query
    search_query: str
    # Grading result
    documents_relevant: bool
    # Number of retrieval attempts
    retrieval_attempts: int


# ============================================================================
# Document Store Setup
# ============================================================================

def initialize_vector_store():
    """
    Initialize the vector store with documents.

    Returns:
        InMemoryVectorStore: Vector store with embedded documents
    """
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # Load documents
    docs = []
    for url in urls:
        docs.extend(WebBaseLoader(url).load())

    # Split into chunks
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

    return vectorstore


# Initialize vector store globally
VECTOR_STORE = initialize_vector_store()


# ============================================================================
# Agent Definitions
# ============================================================================

def create_agent(llm: ChatOpenAI, system_prompt: str):
    """
    Create an agent with a specific system prompt.

    Args:
        llm: The language model to use
        system_prompt: The system prompt defining agent behavior

    Returns:
        Runnable: A chat model bound with the system prompt
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm


def agent_node(state: AgentState, agent, name: str):
    """
    Execute an agent and return the result.

    Args:
        state: Current state
        agent: The agent to execute
        name: Name of the agent (for message attribution)

    Returns:
        dict: Updated state with agent's response
    """
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result.content, name=name)]
    }


# ============================================================================
# Specialized Agents
# ============================================================================

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Researcher Agent - Handles document retrieval
researcher_prompt = """You are a research agent specializing in document retrieval.

Your responsibilities:
- Search through the document collection for relevant information
- Extract the most pertinent passages related to user queries
- Provide comprehensive context from multiple sources when available
- Focus on Lilian Weng's blog posts about AI topics (reward hacking, hallucination, diffusion video)

When asked to research:
1. Formulate effective search queries
2. Retrieve relevant documents
3. Summarize key findings
4. Return structured results

Be thorough but concise in your responses."""

researcher_agent = create_agent(llm, researcher_prompt)


# Grader Agent - Evaluates document relevance
grader_prompt = """You are a document grading specialist.

Your responsibilities:
- Assess whether retrieved documents are relevant to the user's question
- Provide clear yes/no judgments with brief reasoning
- Consider both direct relevance and contextual usefulness
- Be strict but fair in your assessments

Grading criteria:
- Does the document address the question directly?
- Does it provide useful context or related information?
- Is the information specific enough to be helpful?

Respond with: "RELEVANT: [reasoning]" or "NOT RELEVANT: [reasoning]" """

grader_agent = create_agent(llm, grader_prompt)


# Writer Agent - Generates final answers
writer_prompt = """You are an expert writer specializing in clear, accurate answers.

Your responsibilities:
- Synthesize information from provided context into coherent answers
- Cite specific details from the source material
- Maintain accuracy and avoid speculation
- Write in a professional yet accessible tone

Guidelines:
- Base your answer strictly on the provided context
- Keep responses concise (3-6 sentences) unless more detail is needed
- Acknowledge limitations if context is insufficient
- Structure information logically

Your goal is to provide the most helpful and accurate answer possible."""

writer_agent = create_agent(llm, writer_prompt)


# ============================================================================
# Agent Node Functions
# ============================================================================

def researcher_node(state: AgentState):
    """
    Researcher agent node - retrieves relevant documents.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with retrieved documents
    """
    # Get the latest user message
    user_query = None
    for msg in reversed(state["messages"]):
        if not hasattr(msg, "name"):  # Original user message
            user_query = msg.content
            break

    # Use search query if available, otherwise user query
    query = state.get("search_query", user_query)

    # Retrieve documents
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    # Format documents
    doc_content = "\n\n".join([f"Document {i+1}:\n{doc.page_content}"
                               for i, doc in enumerate(docs)])

    # Create response message
    result_msg = f"Retrieved {len(docs)} documents for query: '{query}'\n\n{doc_content[:1000]}..."

    return {
        "messages": [HumanMessage(content=result_msg, name="Researcher")],
        "retrieved_documents": doc_content,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }


def grader_node(state: AgentState):
    """
    Grader agent node - evaluates document relevance.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with grading decision
    """
    # Get user question
    user_query = None
    for msg in reversed(state["messages"]):
        if not hasattr(msg, "name"):
            user_query = msg.content
            break

    docs = state.get("retrieved_documents", "")

    # Create grading prompt
    grading_msg = f"""Please grade these documents for relevance to the question:

Question: {user_query}

Documents:
{docs[:1500]}

Are these documents relevant?"""

    # Execute grader
    result = grader_agent.invoke({
        "messages": [HumanMessage(content=grading_msg)]
    })

    # Parse result
    is_relevant = "RELEVANT:" in result.content.upper() and "NOT RELEVANT:" not in result.content.upper()

    return {
        "messages": [HumanMessage(content=result.content, name="Grader")],
        "documents_relevant": is_relevant
    }


def writer_node(state: AgentState):
    """
    Writer agent node - generates final answer.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with generated answer
    """
    # Get user question
    user_query = None
    for msg in reversed(state["messages"]):
        if not hasattr(msg, "name"):
            user_query = msg.content
            break

    docs = state.get("retrieved_documents", "")

    # Create writing prompt
    writing_msg = f"""Based on the following context, please answer the question:

Question: {user_query}

Context:
{docs}

Provide a clear, concise answer based on this context."""

    # Execute writer
    result = writer_agent.invoke({
        "messages": [HumanMessage(content=writing_msg)]
    })

    return {
        "messages": [HumanMessage(content=result.content, name="Writer")]
    }


# ============================================================================
# Supervisor
# ============================================================================

class RouteResponse(BaseModel):
    """Response model for supervisor routing decisions."""
    next: Literal["Researcher", "Grader", "Writer", "FINISH"]
    reasoning: str


def supervisor_node(state: AgentState):
    """
    Supervisor node that routes to appropriate agents.

    The supervisor analyzes the current state and decides which agent
    should act next based on:
    - Current conversation state
    - Available information
    - Agent capabilities
    - Task requirements

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with routing decision
    """
    # Define available agents and their capabilities
    members = ["Researcher", "Grader", "Writer"]

    # Create supervisor prompt
    system_prompt = f"""You are a supervisor managing a team of AI agents for a RAG system.

Your team:
- Researcher: Retrieves documents from knowledge base
- Grader: Evaluates if retrieved documents are relevant
- Writer: Generates final answers from relevant documents

Current state:
- Retrieved documents: {bool(state.get('retrieved_documents'))}
- Documents graded: {'documents_relevant' in state}
- Documents relevant: {state.get('documents_relevant', 'Not yet graded')}
- Retrieval attempts: {state.get('retrieval_attempts', 0)}

Workflow:
1. Start with Researcher to get documents
2. Use Grader to check relevance
3. If relevant -> Writer for final answer
4. If not relevant and attempts < 2 -> Reformulate and try Researcher again
5. If attempts >= 2 or answer generated -> FINISH

Decide which agent should act next or if we should FINISH."""

    options = members + ["FINISH"]

    # Create routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Given the conversation above, who should act next? Options: {options}"),
    ])

    supervisor_chain = (
        prompt
        | llm.with_structured_output(RouteResponse)
    )

    result = supervisor_chain.invoke(state)

    print(f"\n🎯 Supervisor: {result.next} - {result.reasoning}")

    return {"next": result.next}


# ============================================================================
# Graph Construction
# ============================================================================

def create_graph():
    """
    Create the multi-agent RAG graph.

    Returns:
        CompiledGraph: The compiled LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Grader", grader_node)
    workflow.add_node("Writer", writer_node)

    # Define routing function
    def route(state: AgentState) -> str:
        """Route based on supervisor's decision."""
        return state["next"]

    # Add edges
    workflow.add_edge(START, "Supervisor")

    # Conditional edges from supervisor
    workflow.add_conditional_edges(
        "Supervisor",
        route,
        {
            "Researcher": "Researcher",
            "Grader": "Grader",
            "Writer": "Writer",
            "FINISH": END
        }
    )

    # All agents return to supervisor
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Grader", "Supervisor")
    workflow.add_edge("Writer", "Supervisor")

    # Compile
    return workflow.compile()


# ============================================================================
# Module Export
# ============================================================================

# Create graph instance for LangGraph dev
graph = create_graph()
