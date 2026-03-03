# Agentic RAG Demo for LangGraph Dev

This project demonstrates an agentic RAG (Retrieval-Augmented Generation) system built with LangGraph that can be run with `langgraph dev` for LangSmith tracing and visualization.

## Features

The agentic RAG system includes:

1. **Document Retrieval**: Loads and processes documents from Lilian Weng's blog posts
2. **Intelligent Query Processing**: Decides whether to retrieve documents or respond directly
3. **Document Grading**: Evaluates relevance of retrieved documents
4. **Query Rewriting**: Improves queries when documents are irrelevant
5. **Answer Generation**: Provides concise answers based on relevant context

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 3. Run with LangGraph Dev

To run the demo with LangSmith tracing:

```bash
langgraph dev
```

This will start a development server that you can access at `http://localhost:8123` and will automatically trace all executions in LangSmith.

### 4. Alternative: Run Directly

You can also run the script directly:

```bash
python agentic_rag.py
```

## How It Works

The system follows this workflow:

1. **User Query**: User asks a question
2. **Query Analysis**: The system decides whether to retrieve documents or respond directly
3. **Document Retrieval**: If needed, retrieves relevant documents from the vector store
4. **Relevance Grading**: Evaluates if retrieved documents are relevant to the query
5. **Query Rewriting**: If documents are irrelevant, rewrites the query for better results
6. **Answer Generation**: Provides a final answer based on relevant context

## Example Queries

Try these example queries to see the system in action:

- "What does Lilian Weng say about types of reward hacking?"
- "Hello!"
- "Tell me about hallucination in AI systems"
- "What is diffusion video generation?"

## LangSmith Integration

When running with `langgraph dev`, all executions are automatically traced in LangSmith, allowing you to:

- View the complete execution flow
- Debug individual nodes
- Monitor performance metrics
- Analyze conversation patterns

## Project Structure

```
├── agentic_rag.py          # Main implementation
├── langgraph.json          # LangGraph dev configuration
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Make sure both OpenAI and LangSmith API keys are set in your `.env` file
2. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Network Issues**: The system loads documents from external URLs, so ensure internet connectivity

### Getting Help

If you encounter issues:

1. Check that all environment variables are properly set
2. Verify that all dependencies are installed
3. Ensure you have valid API keys for both OpenAI and LangSmith
4. Check the LangSmith dashboard for detailed execution traces