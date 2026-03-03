#!/usr/bin/env python3
"""
Demo script for the Supervisor-based RAG Agent.

This demonstrates a supervisor agent architecture where specialized subagents
are orchestrated by a supervisor to handle different aspects of the RAG pipeline.
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Run a demonstration of the supervisor-based RAG agent."""
    print("=" * 70)
    print("Supervisor-Based RAG Agent Demo")
    print("=" * 70)
    print("\nArchitecture:")
    print("  🎯 Supervisor Agent - Orchestrates the workflow")
    print("  📚 Retriever Subagent - Retrieves relevant documents")
    print("  ⚖️  Grader Subagent - Evaluates document relevance")
    print("  ✍️  Rewriter Subagent - Reformulates queries")
    print("  💬 Answer Subagent - Generates final answers")
    print("=" * 70)

    # Check if API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in the .env file.")
        return False

    try:
        from src.officebuddy.agents import supervisor_rag_agent

        # Get the graph
        print("\n🔧 Initializing supervisor agent system...")
        graph = supervisor_rag_agent.graph

        # Example queries
        queries = [
            "What does Lilian Weng say about types of reward hacking?",
            "Hello! How are you today?",
            "Tell me about hallucination in AI systems",
            "What techniques are used in diffusion video generation?",
        ]

        print(f"\n🚀 Running {len(queries)} example queries...\n")

        for i, query in enumerate(queries, 1):
            print("\n" + "=" * 70)
            print(f"Query {i}/{len(queries)}: {query}")
            print("-" * 70)

            try:
                # Run the query through the supervisor system
                result = graph.invoke(
                    {
                        "messages": [{"role": "user", "content": query}],
                        "next": "",
                        "retrieval_query": "",
                        "retrieved_docs": "",
                        "is_relevant": False,
                    }
                )

                # Extract and display the final answer
                if result.get("messages"):
                    final_message = result["messages"][-1]
                    if hasattr(final_message, "content"):
                        print("\n✅ Final Answer:")
                        print("-" * 70)
                        print(final_message.content)
                    else:
                        print("\n✅ Response:", final_message)

            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue

            print("=" * 70)

        print("\n✅ Demo completed successfully!")
        print("\n💡 To run with LangGraph dev server:")
        print("   1. Run: langgraph dev")
        print("   2. Open http://localhost:8123")
        print("   3. Select 'supervisor_rag' from available graphs")
        print("\n🔍 Notice how the supervisor orchestrates the subagents!")

        return True

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
