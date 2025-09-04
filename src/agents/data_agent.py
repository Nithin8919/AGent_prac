"""
Data Agent - Fetch data from LlamaIndex
=====================================

This agent handles data fetching from the LlamaIndex connector and processes
the results for the citizen agent.

Features:
- Query execution via LlamaIndex
- Result processing and formatting
- Error handling and validation
- Data aggregation and analysis
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llamaindex_connector import VehicleRegistrationConnector
from citizen_agent import QueryIntent, QueryType

logger = logging.getLogger(__name__)


class DataAgent:
    """Agent for fetching and processing data from the database."""
    
    def __init__(self, db_path: str = "db/vehicles.db", openai_api_key: Optional[str] = None):
        """
        Initialize the data agent.
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key for LlamaIndex
        """
        self.connector = VehicleRegistrationConnector(db_path, openai_api_key)
        self.initialized = False
        
        logger.info("Data Agent initialized")
    
    def initialize(self):
        """Initialize the LlamaIndex connector."""
        try:
            self.connector.initialize()
            self.initialized = True
            logger.info("Data Agent connector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data agent: {str(e)}")
            raise
    
    def execute_query(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """
        Execute a query based on the parsed intent.
        
        Args:
            query_intent: Parsed query intent from citizen agent
            
        Returns:
            Dictionary with query results and metadata
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Data agent not initialized. Call initialize() first.",
                "answer": None,
                "sql_query": None
            }
        
        # Convert intent to natural language question
        question = self._intent_to_question(query_intent)
        
        # Execute query via LlamaIndex
        result = self.connector.query(question)
        
        # Process and enhance the result
        processed_result = self._process_result(result, query_intent)
        
        return processed_result
    
    def _intent_to_question(self, intent: QueryIntent) -> str:
        """Convert query intent back to natural language question."""
        # Get entity from the entities dict for compatibility
        entity = intent.entities.get("entity", "vehicle") if isinstance(intent.entities, dict) else "vehicle"
        
        if intent.query_type == QueryType.COUNT:
            if entity == 'manufacturer':
                return "How many different vehicle manufacturers are there?"
            elif entity == 'fuel_type':
                return "How many different fuel types are used by vehicles?"
            elif entity == 'vehicle_type':
                return "How many different vehicle types are there?"
            else:
                return "How many vehicles are registered in total?"
        
        elif intent.query_type == QueryType.TOP_N:
            limit = intent.limit or 5
            if entity == 'manufacturer':
                return f"What are the top {limit} vehicle manufacturers by registration count?"
            elif entity == 'fuel_type':
                return f"What are the top {limit} fuel types by usage?"
            elif entity == 'vehicle_type':
                return f"What are the top {limit} vehicle types by registration count?"
            else:
                return f"What are the top {limit} most registered items?"
        
        elif intent.query_type == QueryType.COMPARISON:
            if entity == 'fuel_type':
                return "Compare the number of vehicles using petrol vs diesel fuel"
            elif entity == 'vehicle_type':
                return "Compare the number of motorcycles vs cars registered"
            else:
                return "Show a comparison of the data"
        
        elif intent.query_type == QueryType.DISTRIBUTION:
            if entity == 'fuel_type':
                return "What is the distribution of fuel types used by vehicles?"
            elif entity == 'vehicle_type':
                return "What is the distribution of vehicle types?"
            else:
                return "Show the distribution of the data"
        
        elif intent.query_type == QueryType.TREND:
            if 'year' in intent.filters:
                year = intent.filters['year']
                return f"How many vehicles were registered in {year}?"
            else:
                return "What is the trend of vehicle registrations over the years?"
        
        else:
            return "Tell me about vehicle registration data"
    
    def _process_result(self, result: Dict[str, Any], intent: QueryIntent) -> Dict[str, Any]:
        """Process and enhance the query result."""
        if not result.get('success', False):
            return result
        
        answer = result.get('answer', '')
        
        # Enhance answer based on query type
        if intent.query_type == QueryType.COUNT:
            # Try to extract the number from the answer
            import re
            numbers = re.findall(r'\d{1,3}(?:,\d{3})*', answer)
            if numbers:
                # Format the number nicely
                count = numbers[0].replace(',', '')
                if count.isdigit():
                    formatted_count = f"{int(count):,}"
                    answer = answer.replace(numbers[0], formatted_count)
        
        elif intent.query_type == QueryType.TOP_N:
            # Format top-N results nicely
            if 'top' in answer.lower() and 'manufacturer' in answer.lower():
                answer = self._format_top_manufacturers(answer)
            elif 'fuel' in answer.lower():
                answer = self._format_fuel_distribution(answer)
        
        elif intent.query_type == QueryType.COMPARISON:
            # Format comparison results
            answer = self._format_comparison(answer)
        
        result['answer'] = answer
        return result
    
    def _format_top_manufacturers(self, answer: str) -> str:
        """Format top manufacturers result."""
        lines = answer.split('\n')
        formatted_lines = []
        
        for line in lines:
            if any(manufacturer in line.lower() for manufacturer in ['honda', 'hero', 'bajaj', 'maruti', 'tvs']):
                # Extract manufacturer and count
                parts = line.split()
                if len(parts) >= 2:
                    manufacturer = parts[0]
                    count = parts[-1] if parts[-1].replace(',', '').isdigit() else 'N/A'
                    formatted_lines.append(f"• {manufacturer}: {count} registrations")
                else:
                    formatted_lines.append(f"• {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_fuel_distribution(self, answer: str) -> str:
        """Format fuel distribution result."""
        lines = answer.split('\n')
        formatted_lines = []
        
        for line in lines:
            if any(fuel in line.lower() for fuel in ['petrol', 'diesel', 'electric', 'battery']):
                parts = line.split()
                if len(parts) >= 2:
                    fuel_type = parts[0].title()
                    count = parts[-1] if parts[-1].replace(',', '').isdigit() else 'N/A'
                    formatted_lines.append(f"• {fuel_type}: {count} vehicles")
                else:
                    formatted_lines.append(f"• {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_comparison(self, answer: str) -> str:
        """Format comparison result."""
        # Simple formatting for comparison results
        if 'petrol' in answer.lower() and 'diesel' in answer.lower():
            return f"Fuel Type Comparison:\n\n{answer}"
        elif 'motorcycle' in answer.lower() and 'car' in answer.lower():
            return f"Vehicle Type Comparison:\n\n{answer}"
        else:
            return f"Comparison Results:\n\n{answer}"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics."""
        if not self.initialized:
            return {"error": "Data agent not initialized"}
        
        return self.connector.validate_database_connection()
    
    def test_connection(self) -> bool:
        """Test if the data agent can connect to the database."""
        try:
            if not self.initialized:
                self.initialize()
            
            stats = self.get_database_stats()
            return "error" not in stats
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


def main():
    """Test the data agent."""
    print("📊 Testing Data Agent")
    print("=" * 40)
    
    # Test without OpenAI API key first
    agent = DataAgent()
    
    # Test connection
    print("🔍 Testing database connection...")
    if agent.test_connection():
        print("✅ Database connection successful!")
        
        # Get database stats
        stats = agent.get_database_stats()
        if "error" not in stats:
            print(f"   Total records: {stats['total_records']:,}")
            print(f"   Total columns: {stats['total_columns']}")
        
        print("\n💡 To test AI-powered queries, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Then run the full system")
        
    else:
        print("❌ Database connection failed!")


if __name__ == "__main__":
    main()