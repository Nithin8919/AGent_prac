"""
LlamaIndex Connector for Vehicle Registration Database
====================================================

This module sets up LlamaIndex with SQLite database connection for natural language
querying of vehicle registration data.

Features:
- SQLite database connection
- Natural language to SQL query conversion
- Query result processing and formatting
- Error handling and validation
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleRegistrationConnector:
    """Main connector class for LlamaIndex integration with vehicle registration database."""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: Optional[str] = None):
        """
        Initialize the LlamaIndex connector.
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
        """
        self.db_path = Path(db_path)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.sql_database = None
        self.query_engine = None
        self.table_schema_objs = None
        
        logger.info(f"Initialized connector for database: {self.db_path}")
    
    def setup_llm_and_embeddings(self):
        """Set up LLM and embeddings with OpenAI."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Configure LLM
        Settings.llm = OpenAI(
            api_key=self.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Configure embeddings
        Settings.embed_model = OpenAIEmbedding(
            api_key=self.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        logger.info("LLM and embeddings configured successfully")
    
    def connect_to_database(self):
        """Connect to SQLite database and set up SQLDatabase object."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        # Create SQLite connection string for SQLAlchemy
        db_url = f"sqlite:///{self.db_path}"
        
        # Create SQLDatabase object
        self.sql_database = SQLDatabase.from_uri(db_url, include_tables=["vehicle_registrations"])
        
        logger.info("Connected to SQLite database successfully")
        logger.info(f"Available tables: {self.sql_database.get_usable_table_names()}")
    
    def get_table_schema(self) -> str:
        """Get the schema of the vehicle_registrations table."""
        if not self.sql_database:
            raise ValueError("Database not connected. Call connect_to_database() first.")
        
        # Get table schema
        schema = self.sql_database.get_single_table_info("vehicle_registrations")
        logger.info("Retrieved table schema")
        return schema
    
    def setup_query_engine(self):
        """Set up the natural language to SQL query engine."""
        if not self.sql_database:
            raise ValueError("Database not connected. Call connect_to_database() first.")
        
        # Create table schema objects
        table_schema_obj = SQLTableSchema(
            table_name="vehicle_registrations",
            context_str=self.get_table_schema()
        )
        self.table_schema_objs = [table_schema_obj]
        
        # Create query engine
        self.query_engine = NLSQLTableQueryEngine(
            sql_database=self.sql_database,
            tables=["vehicle_registrations"],
            table_schema_objs=self.table_schema_objs,
            verbose=True
        )
        
        logger.info("Query engine set up successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a natural language query against the database.
        
        Args:
            question: Natural language question about vehicle registration data
            
        Returns:
            Dictionary containing query results and metadata
        """
        if not self.query_engine:
            raise ValueError("Query engine not set up. Call setup_query_engine() first.")
        
        try:
            logger.info(f"Executing query: {question}")
            
            # Execute the query
            response = self.query_engine.query(question)
            
            # Extract results
            result = {
                "question": question,
                "answer": str(response),
                "sql_query": getattr(response, 'metadata', {}).get('sql_query', 'N/A'),
                "success": True,
                "error": None
            }
            
            logger.info("Query executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "question": question,
                "answer": None,
                "sql_query": None,
                "success": False,
                "error": error_msg
            }
    
    def get_sample_queries(self) -> List[str]:
        """Get a list of sample queries that can be asked."""
        return [
            "How many vehicles are registered in total?",
            "What are the top 5 vehicle manufacturers by registration count?",
            "How many vehicles use petrol vs diesel fuel?",
            "What is the distribution of vehicle body types?",
            "Which RTO office has the most registrations?",
            "How many vehicles were registered in 2024?",
            "What is the average engine capacity of vehicles?",
            "How many electric vehicles (battery fuel type) are registered?",
            "What are the most popular vehicle models?",
            "How many vehicles have 2 seats vs 5 seats?",
            "Which year had the highest number of registrations?",
            "What percentage of vehicles are motorcycles (solo body type)?",
            "How many vehicles were registered in Hyderabad?",
            "What is the trend of vehicle registrations over the years?",
            "Which manufacturer has the highest market share?"
        ]
    
    def validate_database_connection(self) -> Dict[str, Any]:
        """Validate the database connection and return basic statistics."""
        if not self.sql_database:
            return {"error": "Database not connected"}
        
        try:
            # Get basic database info using SQLAlchemy
            from sqlalchemy import text
            
            with self.sql_database._engine.connect() as conn:
                # Get table count
                result = conn.execute(text("SELECT COUNT(*) FROM vehicle_registrations"))
                total_records = result.scalar()
                
                # Get column info
                result = conn.execute(text("PRAGMA table_info(vehicle_registrations)"))
                columns = result.fetchall()
            
            return {
                "connected": True,
                "total_records": total_records,
                "total_columns": len(columns),
                "columns": [col[1] for col in columns],
                "database_path": str(self.db_path)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def initialize(self):
        """Initialize the complete connector setup."""
        try:
            logger.info("Initializing LlamaIndex connector...")
            
            # Set up LLM and embeddings
            self.setup_llm_and_embeddings()
            
            # Connect to database
            self.connect_to_database()
            
            # Set up query engine
            self.setup_query_engine()
            
            logger.info("LlamaIndex connector initialized successfully!")
            
            # Validate connection
            validation = self.validate_database_connection()
            if "error" not in validation:
                logger.info(f"Database validation successful: {validation['total_records']:,} records")
            
        except Exception as e:
            logger.error(f"Failed to initialize connector: {str(e)}")
            raise


def main():
    """Main function to test the connector."""
    print("🚀 Testing LlamaIndex Connector")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize connector
        connector = VehicleRegistrationConnector()
        connector.initialize()
        
        # Test basic query
        print("\n📊 Testing basic query...")
        result = connector.query("How many vehicles are registered in total?")
        
        if result["success"]:
            print(f"✅ Query successful!")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"SQL: {result['sql_query']}")
        else:
            print(f"❌ Query failed: {result['error']}")
        
        # Show sample queries
        print(f"\n💡 Sample queries you can ask:")
        for i, query in enumerate(connector.get_sample_queries()[:5], 1):
            print(f"   {i}. {query}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()