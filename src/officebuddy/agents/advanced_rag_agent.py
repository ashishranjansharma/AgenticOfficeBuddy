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

# Load environment variables
load_dotenv()

# Configuration
os.environ["USER_AGENT"] = "AdvancedRAGAgent/1.0"


# System Prompts
AGENT_SYSTEM_PROMPT = """You are an intelligent research assistant with access to a document retrieval system.

Your capabilities:
- You can search through Lilian Weng's blog posts about AI topics
- You provide accurate, well-researched answers based on retrieved information
- You cite sources when providing information
- You're honest when you don't have enough information

Guidelines:
- Always use the retrieval tool when asked about specific topics in the blog posts
- Be conversational and helpful
- Provide concise yet comprehensive answers
- If information is not available in the retrieved documents, say so clearly
"""

GRADER_SYSTEM_PROMPT = """You are a document relevance grader.

Your task is to assess whether a retrieved document is relevant to a user's question.

Guidelines:
- Look for keyword matches and semantic relevance
- Consider the overall context and intent of the question
- Be strict but fair in your assessment
- Return 'yes' if the document contains useful information for answering the question
- Return 'no' if the document is off-topic or contains no relevant information
"""

REWRITER_SYSTEM_PROMPT = """You are a query optimization specialist.

Your task is to reformulate user questions to improve document retrieval.

Guidelines:
- Identify the core intent and semantic meaning of the question
- Expand acronyms and clarify ambiguous terms
- Add relevant context that might improve search results
- Keep the reformulated query concise and focused
- Maintain the original question's intent
"""

ANSWER_GENERATOR_SYSTEM_PROMPT = """You are a knowledgeable AI assistant specializing in synthesizing information.

Your task is to generate clear, accurate answers based on retrieved context.

Guidelines:
- Base your answer strictly on the provided context
- Be concise - use 3-5 sentences maximum unless more detail is needed
- Cite specific information from the context when relevant
- If the context doesn't contain enough information, acknowledge this
- Maintain a professional yet friendly tone
"""


# State Definition
class AgentState(TypedDict):
    """State for the advanced RAG agent."""
    messages: Annotated[list, add_messages]
    documents: list
    is_relevant: bool


# Document Processing
def load_and_process_documents():
    """Load documents from URLs and split them into chunks."""
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
    """Create a vector store and retriever from documents."""
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


# Models
def get_agent_model():
    """Get the main agent model with system prompt."""
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_grader_model():
    """Get the grader model."""
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_rewriter_model():
    """Get the query rewriter model."""
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_answer_model():
    """Get the answer generation model."""
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


# Node Functions
def agent_node(state: AgentState):
    """Main agent node that decides whether to retrieve documents or respond directly."""
    retriever_tool = create_retriever()
    model = get_agent_model()

    # Add system prompt if not present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = model.bind_tools([retriever_tool]).invoke(messages)

    return {"messages": [response]}


def grade_documents_node(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """Grade the relevance of retrieved documents."""
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

    # Grade the documents
    class GradeDocuments(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    grader_prompt = f"""Document: {context}

Question: {user_question}

Is this document relevant to answering the question? Consider both keyword matches and semantic meaning."""

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
    """Rewrite the user question to improve retrieval."""
    messages = state["messages"]

    # Find the original user question
    user_question = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    rewriter_prompt = f"""Original question: {user_question}

Please reformulate this question to improve document retrieval. Focus on the key concepts and intent."""

    model = get_rewriter_model()
    rewriter_messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=rewriter_prompt)
    ]

    response = model.invoke(rewriter_messages)

    # Return the rewritten question as a new user message
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer_node(state: AgentState):
    """Generate the final answer based on relevant documents."""
    messages = state["messages"]

    # Extract question and context
    user_question = None
    context = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        if hasattr(msg, 'content') and msg != messages[0]:  # Skip system message
            context = msg.content

    answer_prompt = f"""Question: {user_question}

Context: {context}

Please provide a clear, concise answer based on the context above."""

    model = get_answer_model()
    answer_messages = [
        SystemMessage(content=ANSWER_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=answer_prompt)
    ]

    response = model.invoke(answer_messages)

    return {"messages": [response]}


# Graph Construction
def create_agent_graph():
    """Create and compile the LangGraph workflow."""
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

    # Compile the graph
    graph = workflow.compile()
    return graph


# Create the graph instance for LangGraph dev
graph = create_agent_graph()
