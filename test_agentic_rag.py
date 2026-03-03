#!/usr/bin/env python3
"""
Test script for the Agentic RAG implementation.
This script tests the basic functionality without requiring API keys.
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from src.officebuddy.agents import agentic_rag  # noqa: F401
        print("✓ Successfully imported agentic_rag module")
        return True
    except ImportError as e:
        print(f"✗ Failed to import agentic_rag: {e}")
        return False

def test_functions():
    """Test that all required functions exist."""
    try:
        from src.officebuddy.agents import agentic_rag

        # Check if all required functions exist
        required_functions = [
            'load_and_process_documents',
            'create_retriever',
            'generate_query_or_respond',
            'grade_documents',
            'rewrite_question',
            'generate_answer',
            'create_workflow',
        ]
        
        for func_name in required_functions:
            if hasattr(agentic_rag, func_name):
                print(f"✓ Function {func_name} exists")
            else:
                print(f"✗ Function {func_name} missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing functions: {e}")
        return False

def test_document_loading():
    """Test document loading functionality."""
    try:
        from src.officebuddy.agents import agentic_rag

        # Set a dummy USER_AGENT to avoid warnings
        os.environ["USER_AGENT"] = "TestAgent"

        print("Testing document loading...")
        docs = agentic_rag.load_and_process_documents()
        
        if docs and len(docs) > 0:
            print(f"✓ Successfully loaded {len(docs)} document chunks")
            print(f"✓ First chunk preview: {docs[0].page_content[:100]}...")
            return True
        else:
            print("✗ No documents loaded")
            return False
            
    except Exception as e:
        print(f"✗ Error testing document loading: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Agentic RAG Tests")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Function Test", test_functions), 
        ("Document Loading Test", test_document_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print(f"\n{'=' * 40}")
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The agentic RAG implementation is ready.")
        print("\nTo run with LangGraph dev:")
        print("1. Set up your .env file with API keys")
        print("2. Run: langgraph dev")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)