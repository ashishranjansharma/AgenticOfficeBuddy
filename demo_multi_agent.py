#!/usr/bin/env python3
"""
Demo script for Multi-Agent RAG System.

Demonstrates the LangGraph supervisor pattern with specialized agents:
- Supervisor: Orchestrates the workflow
- Researcher: Retrieves relevant documents
- Grader: Evaluates document quality
- Writer: Generates final answers
"""

import os
import sys
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Run demonstration of the multi-agent RAG system."""
    print("=" * 80)
    print(" " * 20 + "Multi-Agent RAG System Demo")
    print("=" * 80)
    print("\n📋 Agent Team:")
    print("  🎯 Supervisor  - Orchestrates workflow and routing decisions")
    print("  🔍 Researcher  - Retrieves documents from knowledge base")
    print("  ⚖️  Grader     - Evaluates document relevance and quality")
    print("  ✍️  Writer     - Generates clear, accurate answers")
    print("\n" + "=" * 80)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Error: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in the .env file.")
        return False

    try:
        from src.officebuddy.agents import multi_agent_rag

        print("\n🔧 Initializing multi-agent system...")
        graph = multi_agent_rag.graph

        # Test queries
        queries = [
            "What are the main types of reward hacking?",
            "Explain hallucination in AI systems",
            "Hello, how are you?",
            "What challenges exist in diffusion video generation?",
        ]

        print(f"\n🚀 Running {len(queries)} test queries...\n")

        for i, query in enumerate(queries, 1):
            print("\n" + "=" * 80)
            print(f"Query {i}/{len(queries)}: {query}")
            print("-" * 80)

            try:
                # Initialize state
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "next": "",
                    "retrieved_documents": "",
                    "search_query": "",
                    "documents_relevant": False,
                    "retrieval_attempts": 0
                }

                # Run the graph
                result = graph.invoke(initial_state)

                # Display final answer
                if result.get("messages"):
                    print("\n📝 Conversation Flow:")
                    print("-" * 80)
                    for msg in result["messages"]:
                        if hasattr(msg, "name"):
                            print(f"\n[{msg.name}]: {msg.content[:200]}...")
                        else:
                            print(f"\n[User]: {msg.content}")

                    # Extract final answer
                    final_msg = result["messages"][-1]
                    if hasattr(final_msg, "name") and final_msg.name == "Writer":
                        print("\n" + "=" * 80)
                        print("✅ FINAL ANSWER:")
                        print("-" * 80)
                        print(final_msg.content)
                        print("=" * 80)

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("\n💡 To run with LangGraph Studio:")
        print("   1. Run: langgraph dev")
        print("   2. Open: http://localhost:8123")
        print("   3. Select: 'multi_agent_rag'")
        print("\n🎓 This follows the official LangGraph supervisor pattern!")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
