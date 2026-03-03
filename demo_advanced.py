#!/usr/bin/env python3
"""
Demo script for the Advanced RAG Agent.
This script demonstrates the enhanced agent with system prompts and better architecture.
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Run a demonstration of the advanced RAG agent."""
    print("Advanced RAG Agent Demo")
    print("=" * 60)
    print("This demo shows the enhanced agent with:")
    print("1. Configurable system prompts for different roles")
    print("2. Proper agent-based architecture")
    print("3. Document relevance grading")
    print("4. Intelligent query rewriting")
    print("5. Context-aware answer generation")
    print("=" * 60)

    # Check if API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in the .env file or environment variables.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return False

    try:
        from src.officebuddy.agents import advanced_rag_agent

        # Get the graph
        print("\nInitializing advanced agent...")
        graph = advanced_rag_agent.graph

        # Example queries
        queries = [
            "What does Lilian Weng say about types of reward hacking?",
            "Tell me about hallucination in AI systems",
            "Hello! How are you?",
            "What are the main challenges in diffusion video generation?",
        ]

        print(f"\nRunning {len(queries)} example queries...")
        print("=" * 60)

        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print("-" * 60)

            try:
                # Stream the execution
                for chunk in graph.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query,
                            }
                        ]
                    },
                    stream_mode="values"
                ):
                    if "messages" in chunk and chunk["messages"]:
                        last_message = chunk["messages"][-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            # Only print AI responses
                            if hasattr(last_message, 'type') and last_message.type == 'ai':
                                print("\n🤖 Agent Response:")
                                print(last_message.content)

            except Exception as e:
                print(f"❌ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 60)
        print("\n✅ Demo completed successfully!")
        print("\nTo run with LangGraph dev server:")
        print("1. Set up your .env file with OpenAI and LangSmith API keys")
        print("2. Run: langgraph dev")
        print("3. Open http://localhost:8123 in your browser")
        print("4. Select 'advanced_rag' from the available graphs")

        return True

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
