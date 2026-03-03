"""
Supervisor-Based RAG Agent with LangGraph

This implementation uses a supervisor agent pattern where:
1. Supervisor Agent: Orchestrates the workflow and delegates tasks to subagents
2. Retriever Subagent: Handles document retrieval operations
3. Grader Subagent: Evaluates document relevance
4. Rewriter Subagent: Reformulates queries for better retrieval
5. Answer Subagent: Generates final answers from context

Architecture:
    User Query -> Supervisor -> [Retriever | Direct Answer]
                     ↓
                  Grader -> [Answer | Rewriter]
                     ↓              ↓
                  Answer      -> Supervisor
"""

import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

# Import prompts
from .prompts import PromptTemplates

# ============================================================================
# Environment Setup
# ============================================================================

load_dotenv()
os.environ["USER_AGENT"] = "SupervisorRAGAgent/1.0"


# ============================================================================
# Supervisor and Subagent Prompts
# ============================================================================

SUPERVISOR_PROMPT = """You are a supervisor agent managing a team of specialized subagents for a RAG system.

Your team consists of:
1. Retriever: Retrieves relevant documents from the knowledge base
2. Grader: Evaluates if retrieved documents are relevant
3. Rewriter: Reformulates queries to improve retrieval
4. Answer: Generates final answers based on context

Your responsibilities:
- Analyze user queries and determine the best course of action
- Delegate tasks to appropriate subagents
- Ensure smooth workflow between subagents
- Provide direct answers for simple queries that don't need retrieval
- Maintain conversation context and quality

Decision Guidelines:
- For questions about specific topics in the blog posts -> use Retriever
- For general greetings or chitchat -> respond directly with FINISH
- Always ensure documents are graded before generating answers
- If documents are irrelevant, route to Rewriter for query improvement
"""

RETRIEVER_SUBAGENT_PROMPT = """You are a retriever subagent specialized in document search.

Your responsibilities:
- Search through Lilian Weng's blog posts on AI topics
- Retrieve the most relevant documents for user queries
- Return document content to the supervisor
- Focus on precision and relevance

Topics you can search:
- Reward hacking in reinforcement learning
- Hallucination in AI systems
- Diffusion video generation
- Related AI/ML concepts
"""

GRADER_SUBAGENT_PROMPT = """You are a grader subagent specialized in document relevance assessment.

Your responsibilities:
- Evaluate if retrieved documents are relevant to the user's question
- Provide binary yes/no decisions on relevance
- Consider both keyword matches and semantic meaning
- Be strict but fair in your assessment

Decision criteria:
- YES: Document contains information that helps answer the question
- NO: Document is off-topic or lacks relevant information
"""

REWRITER_SUBAGENT_PROMPT = """You are a rewriter subagent specialized in query optimization.

Your responsibilities:
- Reformulate user questions to improve document retrieval
- Identify core intent and semantic meaning
- Expand acronyms and clarify ambiguous terms
- Maintain the original question's intent while improving searchability

Techniques:
- Add relevant context and keywords
- Clarify technical terms
- Break down complex questions
- Focus on key concepts
"""

ANSWER_SUBAGENT_PROMPT = """You are an answer generation subagent specialized in synthesizing information.

Your responsibilities:
- Generate clear, accurate answers based on retrieved context
- Cite specific information from the context when relevant
- Maintain a professional yet friendly tone
- Be concise (3-5 sentences) unless more detail is needed

Quality guidelines:
- Base answers strictly on provided context
- Acknowledge if context is insufficient
- Provide well-structured responses
- Maintain factual accuracy
"""


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    State for the supervisor-based RAG system.

    Attributes:
        messages: Conversation history
        next: Next agent to call
        retrieval_query: Query to use for retrieval
        retrieved_docs: Documents retrieved from vector store
        is_relevant: Whether retrieved docs are relevant
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    retrieval_query: str
    retrieved_docs: str
    is_relevant: bool


# ============================================================================
# Document Processing Functions
# ============================================================================

def load_and_process_documents():
    """
    Load documents from web URLs and process them into chunks.

    Returns:
        list: List of processed document chunks ready for embedding
    """
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=75
    )
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


def create_retriever():
    """
    Create a vector store and retriever tool from processed documents.

    Returns:
        Tool: A retriever tool for document search
    """
    doc_splits = load_and_process_documents()

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on AI topics.",
    )

    return retriever_tool


# Initialize retriever once
RETRIEVER_TOOL = create_retriever()


# ============================================================================
# Subagent Implementations
# ============================================================================

def supervisor_agent(state: AgentState):
    """
    Supervisor agent that orchestrates the workflow.

    Analyzes the current state and decides which subagent should act next.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with next agent decision
    """
    messages = state["messages"]

    # Define routing options
    class RouteDecision(BaseModel):
        """Decision on which agent to route to next."""
        next: Literal["retriever", "grader", "rewriter", "answer", "FINISH"] = Field(
            description="The next agent to call: retriever, grader, rewriter, answer, or FINISH"
        )
        reasoning: str = Field(
            description="Brief explanation of why this route was chosen"
        )

    # Prepare context for supervisor
    context_info = f"""
    Current state:
    - Has retrieval query: {bool(state.get('retrieval_query'))}
    - Has retrieved docs: {bool(state.get('retrieved_docs'))}
    - Docs are relevant: {state.get('is_relevant', False)}
    - Number of messages: {len(messages)}
    """

    model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RouteDecision)

    supervisor_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages,
        HumanMessage(content=f"{context_info}\n\nWhat should be the next step?")
    ]

    response = model.invoke(supervisor_messages)

    print(f"🎯 Supervisor decision: {response.next} - {response.reasoning}")

    return {"next": response.next}


def retriever_subagent(state: AgentState):
    """
    Retriever subagent that handles document retrieval.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with retrieved documents
    """
    messages = state["messages"]

    # Get the latest user question
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # Use retrieval query if available, otherwise use user question
    query = state.get("retrieval_query", user_question)

    # Retrieve documents
    retrieved_docs = RETRIEVER_TOOL.invoke({"query": query})

    print(f"📚 Retriever: Retrieved docs for query: '{query}'")

    return {
        "retrieved_docs": retrieved_docs,
        "messages": [SystemMessage(content=f"Retrieved documents: {retrieved_docs[:500]}...")]
    }


def grader_subagent(state: AgentState):
    """
    Grader subagent that evaluates document relevance.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with relevance decision
    """
    messages = state["messages"]
    retrieved_docs = state.get("retrieved_docs", "")

    # Get user question
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("Retrieved"):
            user_question = msg.content
            break

    # Grade documents
    class GradeDecision(BaseModel):
        """Document relevance grading."""
        is_relevant: bool = Field(description="True if documents are relevant, False otherwise")
        reasoning: str = Field(description="Explanation of the grading decision")

    grader_prompt = PromptTemplates.grader_prompt(retrieved_docs, user_question)

    model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(GradeDecision)
    grader_messages = [
        SystemMessage(content=GRADER_SUBAGENT_PROMPT),
        HumanMessage(content=grader_prompt)
    ]

    response = model.invoke(grader_messages)

    print(f"⚖️ Grader: Documents are {'relevant' if response.is_relevant else 'not relevant'} - {response.reasoning}")

    return {
        "is_relevant": response.is_relevant,
        "messages": [SystemMessage(content=f"Grader assessment: {response.reasoning}")]
    }


def rewriter_subagent(state: AgentState):
    """
    Rewriter subagent that reformulates queries.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with rewritten query
    """
    messages = state["messages"]

    # Get original question
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("Retrieved"):
            user_question = msg.content
            break

    # Rewrite query
    rewriter_prompt = PromptTemplates.rewriter_prompt(user_question)

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    rewriter_messages = [
        SystemMessage(content=REWRITER_SUBAGENT_PROMPT),
        HumanMessage(content=rewriter_prompt)
    ]

    response = model.invoke(rewriter_messages)
    rewritten_query = response.content

    print(f"✍️ Rewriter: '{user_question}' -> '{rewritten_query}'")

    return {
        "retrieval_query": rewritten_query,
        "messages": [SystemMessage(content=f"Rewritten query: {rewritten_query}")]
    }


def answer_subagent(state: AgentState):
    """
    Answer subagent that generates final answers.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with generated answer
    """
    messages = state["messages"]
    retrieved_docs = state.get("retrieved_docs", "")

    # Get user question
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("Retrieved"):
            user_question = msg.content
            break

    # Generate answer
    answer_prompt = PromptTemplates.answer_prompt(user_question, retrieved_docs)

    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    answer_messages = [
        SystemMessage(content=ANSWER_SUBAGENT_PROMPT),
        HumanMessage(content=answer_prompt)
    ]

    response = model.invoke(answer_messages)

    print("💬 Answer: Generated response")

    return {"messages": [response]}


# ============================================================================
# Graph Construction
# ============================================================================

def create_supervisor_graph():
    """
    Create and compile the supervisor-based LangGraph workflow.

    Returns:
        CompiledGraph: Executable LangGraph workflow

    Workflow:
        START -> supervisor -> [retriever | answer | FINISH]
                     ↓              ↓
                  grader      -> supervisor
                     ↓
              [answer | rewriter]
                  ↓         ↓
               FINISH -> supervisor
    """
    workflow = StateGraph(AgentState)

    # Add all subagent nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("retriever", retriever_subagent)
    workflow.add_node("grader", grader_subagent)
    workflow.add_node("rewriter", rewriter_subagent)
    workflow.add_node("answer", answer_subagent)

    # Define routing logic
    def route_supervisor(state: AgentState) -> str:
        """Route based on supervisor's decision."""
        return state["next"]

    # Start with supervisor
    workflow.add_edge(START, "supervisor")

    # Supervisor routes to subagents or finishes
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "retriever": "retriever",
            "grader": "grader",
            "rewriter": "rewriter",
            "answer": "answer",
            "FINISH": END,
        }
    )

    # After retrieval, go back to supervisor
    workflow.add_edge("retriever", "supervisor")

    # After grading, go back to supervisor
    workflow.add_edge("grader", "supervisor")

    # After rewriting, go back to supervisor
    workflow.add_edge("rewriter", "supervisor")

    # After answering, go back to supervisor (who will likely FINISH)
    workflow.add_edge("answer", "supervisor")

    # Compile the graph
    graph = workflow.compile()
    return graph


# ============================================================================
# Module Exports
# ============================================================================

# Create the graph instance for LangGraph dev server
graph = create_supervisor_graph()
