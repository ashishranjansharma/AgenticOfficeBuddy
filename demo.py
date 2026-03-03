#!/usr/bin/env python3
"""
Demo script for the Agentic RAG system.
This script demonstrates how to use the agentic RAG system with example queries.
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_demo():
    """Run a demonstration of the agentic RAG system."""
    print("Agentic RAG Demo")
    print("=" * 50)
    print("This demo shows how the agentic RAG system works.")
    print("The system will:")
    print("1. Analyze your query")
    print("2. Decide whether to retrieve documents")
    print("3. Grade document relevance")
    print("4. Rewrite queries if needed")
    print("5. Generate final answers")
    print("=" * 50)
    
    # Check if API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in the .env file or environment variables.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        from src.officebuddy.agents import agentic_rag

        # Create the workflow
        print("Creating workflow...")
        graph = agentic_rag.graph
        
        # Example queries
        queries = [
            "What does Lilian Weng say about types of reward hacking?",
            "Hello! How are you?",
            "Tell me about hallucination in AI systems",
            "What is diffusion video generation?",
        ]
        
        print(f"\nRunning {len(queries)} example queries...")
        print("=" * 50)
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
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
                    }
                ):
                    for node, update in chunk.items():
                        print(f"\n🔄 Node: {node}")
                        if "messages" in update and update["messages"]:
                            message = update["messages"][-1]
                            if hasattr(message, 'pretty_print'):
                                message.pretty_print()
                            else:
                                print(f"Content: {message.content}")
                        print()
                        
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                continue
            
            print("=" * 50)
        
        print("\n✅ Demo completed successfully!")
        print("\nTo run with LangGraph dev for full tracing:")
        print("1. Set up your .env file with both OpenAI and LangSmith API keys")
        print("2. Run: langgraph dev")
        print("3. Open http://localhost:8123 in your browser")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)