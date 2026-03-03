"""
Advanced LangGraph-based RAG Agent with System Prompts

This implementation provides an enhanced agentic RAG system with:
1. Configurable system prompts for different agent roles
2. Proper agent-based architecture using LangGraph
3. Document retrieval with relevance grading
4. Query rewriting for improved results
5. Context-aware answer generation
"""

import os
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv

# Import system prompts from external configuration
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    PromptTemplates,
)

# ============================================================================
# Environment Setup
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure user agent for web scraping
os.environ["USER_AGENT"] = "AdvancedRAGAgent/1.0"


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """
    State schema for the advanced RAG agent.

    Attributes:
        messages: List of conversation messages with automatic message handling
        documents: Retrieved documents from the vector store
        is_relevant: Boolean indicating if retrieved documents are relevant
    """
    messages: Annotated[list, add_messages]
    documents: list
    is_relevant: bool


# ============================================================================
# Document Processing Functions
# ============================================================================

def load_and_process_documents():
    """
    Load documents from web URLs and process them into chunks.

    This function:
    1. Loads blog posts from specified URLs using WebBaseLoader
    2. Flattens the list of loaded documents
    3. Splits documents into chunks using RecursiveCharacterTextSplitter
    4. Uses tiktoken encoding for accurate token counting

    Returns:
        list: List of processed document chunks ready for embedding

    Note:
        - Chunk size: 250 tokens with 75 token overlap
        - URLs point to Lilian Weng's blog posts on AI topics
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

    This function:
    1. Loads and processes documents into chunks
    2. Creates an in-memory vector store with OpenAI embeddings
    3. Configures retriever to return top 4 most relevant documents
    4. Wraps retriever in a LangChain tool for agent use

    Returns:
        Tool: A retriever tool that can be bound to the agent for document search

    Note:
        Uses OpenAI's text-embedding-ada-002 model for embeddings
    """
    doc_splits = load_and_process_documents()

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on AI topics including reward hacking, hallucination, and diffusion video generation.",
    )

    return retriever_tool


# ============================================================================
# Model Initialization Functions
# ============================================================================

def get_agent_model():
    """
    Initialize the main agent model.

    Returns:
        ChatOpenAI: GPT-4o model with temperature=0 for deterministic responses
    """
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_grader_model():
    """
    Initialize the document grader model.

    Returns:
        ChatOpenAI: GPT-4o model with temperature=0 for consistent grading
    """
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_rewriter_model():
    """
    Initialize the query rewriter model.

    Returns:
        ChatOpenAI: GPT-4o model with temperature=0 for deterministic rewrites
    """
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_answer_model():
    """
    Initialize the answer generation model.

    Returns:
        ChatOpenAI: GPT-4o model with temperature=0.3 for slightly creative answers
    """
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


# ============================================================================
# Graph Node Functions
# ============================================================================

def agent_node(state: AgentState):
    """
    Main agent node that decides whether to retrieve documents or respond directly.

    This node:
    1. Adds system prompt to message history if not present
    2. Binds the retriever tool to the model
    3. Invokes the model to decide on tool use or direct response

    Args:
        state: Current agent state containing messages and metadata

    Returns:
        dict: Updated state with agent's response message
    """
    retriever_tool = create_retriever()
    model = get_agent_model()

    # Add system prompt if not present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = model.bind_tools([retriever_tool]).invoke(messages)

    return {"messages": [response]}


def grade_documents_node(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Grade the relevance of retrieved documents to the user's question.

    This node:
    1. Extracts the user question from message history
    2. Gets the retrieved document context
    3. Uses a grader model to assess relevance
    4. Returns routing decision based on relevance score

    Args:
        state: Current agent state with messages and retrieved documents

    Returns:
        str: Either "generate_answer" if relevant, or "rewrite_question" if not

    Note:
        Uses structured output to ensure consistent binary yes/no grading
    """
    messages = state["messages"]

    # Extract the user question and retrieved context
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # Get the last tool message (retrieved documents)
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        context = last_message.content
    else:
        return "rewrite_question"

    # Grade the documents using structured output
    class GradeDocuments(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    grader_prompt = PromptTemplates.grader_prompt(context, user_question)

    model = get_grader_model().with_structured_output(GradeDocuments)
    grader_messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=grader_prompt)
    ]

    response = model.invoke(grader_messages)

    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def rewrite_question_node(state: AgentState):
    """
    Rewrite the user question to improve document retrieval.

    This node:
    1. Extracts the original user question
    2. Uses a rewriter model to reformulate the query
    3. Returns the improved question for another retrieval attempt

    Args:
        state: Current agent state with message history

    Returns:
        dict: Updated state with rewritten question as new user message

    Note:
        Helps recover from failed retrievals by reformulating queries
    """
    messages = state["messages"]

    # Find the original user question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    rewriter_prompt = PromptTemplates.rewriter_prompt(user_question)

    model = get_rewriter_model()
    rewriter_messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=rewriter_prompt)
    ]

    response = model.invoke(rewriter_messages)

    # Return the rewritten question as a new user message
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer_node(state: AgentState):
    """
    Generate the final answer based on relevant documents.

    This node:
    1. Extracts the user question and retrieved context
    2. Uses an answer generation model with specific system prompt
    3. Synthesizes a clear, concise answer from the context

    Args:
        state: Current agent state with question and retrieved documents

    Returns:
        dict: Updated state with generated answer message

    Note:
        Uses temperature=0.3 for slightly more natural responses
    """
    messages = state["messages"]

    # Extract question and context
    user_question = None
    context = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        if hasattr(msg, 'content') and msg != messages[0]:  # Skip system message
            context = msg.content

    answer_prompt = PromptTemplates.answer_prompt(user_question, context)

    model = get_answer_model()
    answer_messages = [
        SystemMessage(content=ANSWER_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=answer_prompt)
    ]

    response = model.invoke(answer_messages)

    return {"messages": [response]}


# ============================================================================
# Graph Construction
# ============================================================================

def create_agent_graph():
    """
    Create and compile the LangGraph workflow for the RAG agent.

    This function:
    1. Initializes the retriever tool
    2. Creates a StateGraph with all necessary nodes
    3. Defines conditional edges for routing logic
    4. Compiles the graph into an executable workflow

    Returns:
        CompiledGraph: Executable LangGraph workflow

    Workflow:
        START -> agent -> [tools | END]
                   ↓
              retrieve -> grade_documents -> [generate_answer | rewrite_question]
                                                    ↓                ↓
                                                   END          -> agent

    Note:
        The graph implements adaptive RAG with query rewriting on failed retrieval
    """
    retriever_tool = create_retriever()

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Add edges
    workflow.add_edge(START, "agent")

    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Conditional edge from retrieve
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_node,
    )

    # Edges for answer generation and question rewriting
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "agent")

    # Compile the graph into executable workflow
    graph = workflow.compile()
    return graph


# ============================================================================
# Module Exports
# ============================================================================

# Create the graph instance for LangGraph dev server
# This is the main entry point when running with `langgraph dev`
graph = create_agent_graph()
