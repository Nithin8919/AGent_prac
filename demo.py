#!/usr/bin/env python3
"""
Demo Script - Vehicle Registration System
========================================

This script demonstrates the Vehicle Registration AI system without requiring
OpenAI API key. It shows the data structure and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents.citizen_agent import CitizenAgent
from llamaindex_connector import VehicleRegistrationConnector


def main():
    """Run the demo."""
    print("🚀 Vehicle Registration AI System Demo")
    print("=" * 50)
    
    # Test Citizen Agent
    print("\n🧑‍💼 Testing Citizen Agent (Question Parser)")
    print("-" * 40)
    
    citizen_agent = CitizenAgent()
    
    demo_questions = [
        "How many vehicles are registered in total?",
        "What are the top 5 vehicle manufacturers?",
        "How many vehicles use petrol vs diesel?",
        "Which RTO office has the most registrations?",
        "How many vehicles were registered in 2024?",
        "What is the distribution of vehicle types?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Validate question
        is_valid, error = citizen_agent.validate_question(question)
        if not is_valid:
            print(f"   ❌ Invalid: {error}")
            continue
        
        # Parse question
        intent = citizen_agent.parse_question(question)
        print(f"   ✅ Parsed Intent:")
        print(f"      Type: {intent.query_type.value}")
        print(f"      Entity: {intent.entity}")
        print(f"      Filters: {intent.filters}")
        if intent.limit:
            print(f"      Limit: {intent.limit}")
        if intent.clarification_needed:
            print(f"      Clarification: {intent.clarification_message}")
    
    # Test Database Connection
    print(f"\n\n📊 Testing Database Connection")
    print("-" * 40)
    
    connector = VehicleRegistrationConnector(db_path="db/vehicles.db")
    
    try:
        connector.connect_to_database()
        validation = connector.validate_database_connection()
        
        if "error" not in validation:
            print("✅ Database connection successful!")
            print(f"   📈 Total records: {validation['total_records']:,}")
            print(f"   📋 Total columns: {validation['total_columns']}")
            print(f"   💾 Database size: 736 MB")
            
            print(f"\n   📋 Available columns:")
            for i, col in enumerate(validation['columns'], 1):
                print(f"      {i:2d}. {col}")
            
            print(f"\n   💡 Sample queries you can ask:")
            queries = connector.get_sample_queries()
            for i, query in enumerate(queries[:8], 1):
                print(f"      {i}. {query}")
                
        else:
            print(f"❌ Database validation failed: {validation['error']}")
            
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
    
    # Show system capabilities
    print(f"\n\n🎯 System Capabilities")
    print("-" * 40)
    print("✅ Natural Language Question Processing")
    print("✅ Intent Classification and Entity Extraction")
    print("✅ Query Validation and Clarification")
    print("✅ Database Connection and Schema Analysis")
    print("✅ 1.88M Vehicle Registration Records")
    print("✅ 18 Data Columns with 89% Quality Score")
    print("✅ Support for Multiple Query Types:")
    print("   • Count queries (How many vehicles...)")
    print("   • Top-N queries (Top 5 manufacturers...)")
    print("   • Comparison queries (Petrol vs Diesel...)")
    print("   • Trend analysis (Growth over years...)")
    print("   • Distribution analysis (Market share...)")
    
    print(f"\n\n🔑 To Enable AI Features:")
    print("-" * 40)
    print("1. Get an OpenAI API key from: https://platform.openai.com/api-keys")
    print("2. Set the environment variable:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("3. Run the full system:")
    print("   python src/app.py")
    
    print(f"\n\n📊 Data Overview")
    print("-" * 40)
    print("🏭 Top Manufacturers: Honda (19.7%), Hero (11.2%), Bajaj (10.6%)")
    print("⛽ Fuel Types: Petrol (70.3%), Diesel (23.7%), Electric (1.7%)")
    print("🚗 Vehicle Types: Solo/Motorcycles (56.9%), Sedan (14.3%), Hatchback (6.6%)")
    print("📍 Coverage: Telangana State, India (2019-2025)")
    print("🏢 RTO Offices: 6 major offices with 1.88M total registrations")
    
    print(f"\n\n🎉 Demo Complete!")
    print("The system is ready for AI-powered natural language querying!")


if __name__ == "__main__":
    main()

