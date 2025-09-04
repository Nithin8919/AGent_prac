#!/usr/bin/env python3
"""
Test script for LlamaIndex connector
===================================

This script tests the LlamaIndex connector without requiring OpenAI API key.
It validates the database connection and shows sample queries.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llamaindex_connector import VehicleRegistrationConnector


def test_database_connection():
    """Test database connection without LLM setup."""
    print("🔍 Testing Database Connection")
    print("=" * 40)
    
    try:
        # Initialize connector (without LLM setup)
        connector = VehicleRegistrationConnector()
        
        # Connect to database
        connector.connect_to_database()
        
        # Validate connection
        validation = connector.validate_database_connection()
        
        if "error" not in validation:
            print("✅ Database connection successful!")
            print(f"   Total records: {validation['total_records']:,}")
            print(f"   Total columns: {validation['total_columns']}")
            print(f"   Database path: {validation['database_path']}")
            
            print(f"\n📋 Available columns:")
            for i, col in enumerate(validation['columns'], 1):
                print(f"   {i:2d}. {col}")
            
            return True
        else:
            print(f"❌ Database validation failed: {validation['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False


def show_sample_queries():
    """Show sample queries that can be asked."""
    print(f"\n💡 Sample Queries for Vehicle Registration Data")
    print("=" * 50)
    
    connector = VehicleRegistrationConnector()
    queries = connector.get_sample_queries()
    
    for i, query in enumerate(queries, 1):
        print(f"{i:2d}. {query}")


def main():
    """Main test function."""
    print("🚀 LlamaIndex Connector Test")
    print("=" * 50)
    
    # Test database connection
    if test_database_connection():
        # Show sample queries
        show_sample_queries()
        
        print(f"\n🔑 To use AI-powered queries, set your OpenAI API key:")
        print(f"   export OPENAI_API_KEY='your-api-key-here'")
        print(f"   Then run: python src/llamaindex_connector.py")
    else:
        print(f"\n❌ Please fix database connection issues first.")


if __name__ == "__main__":
    main()
