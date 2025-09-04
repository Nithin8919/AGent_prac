"""
Main Application - Intelligent Vehicle Registration Chatbot
==========================================================

This is the main entry point for the Vehicle Registration AI system.
It uses an intelligent citizen agent that can handle any question and provide
comprehensive answers about Telangana vehicle registration data.

Features:
- Intelligent question interpretation and rewriting
- AI-powered data querying with LlamaIndex
- Comprehensive answers with context
- Handles any type of question intelligently
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agents.citizen_agent import IntelligentCitizenAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntelligentVehicleRegistrationApp:
    """Main application class for the intelligent Vehicle Registration system."""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: Optional[str] = None):
        """
        Initialize the application.
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key for AI features
        """
        self.db_path = db_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize the intelligent citizen agent
        self.citizen_agent = IntelligentCitizenAgent()
        
        self.initialized = False
        
        logger.info("Intelligent Vehicle Registration App initialized")
    
    def initialize(self):
        """Initialize the system."""
        try:
            logger.info("Initializing Intelligent Vehicle Registration App...")
            
            # The citizen agent initializes its own connector
            # No additional initialization needed
            
            self.initialized = True
            logger.info("✅ System initialized successfully!")
            
            # Show system status
            self.show_system_status()
            
        except Exception as e:
            logger.error(f"Failed to initialize app: {str(e)}")
            raise
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process any question through the intelligent pipeline.
        
        Args:
            question: Any natural language question
            
        Returns:
            Dictionary with comprehensive response and metadata
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "System not initialized. Call initialize() first.",
                "answer": None
            }
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Use the intelligent citizen agent to process any question
            result = self.citizen_agent.process_question(question)
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "answer": None
            }
    
    def get_suggested_questions(self) -> list:
        """Get a list of suggested questions citizens can ask."""
        return self.citizen_agent.get_suggested_questions()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        info = {
            "initialized": self.initialized,
            "database_path": self.db_path,
            "has_openai_key": bool(self.openai_api_key),
            "system_type": "Intelligent AI-Powered",
            "capabilities": [
                "Handles any type of question",
                "Intelligent question rewriting",
                "Comprehensive answers with context",
                "Real-time data from 1.88M records",
                "AI-powered insights and analysis"
            ]
        }
        
        if self.initialized:
            # Get database stats from the connector
            try:
                db_stats = self.citizen_agent.connector.validate_database_connection()
                if db_stats.get("connected"):
                    info["database"] = {
                        "total_records": db_stats["total_records"],
                        "total_columns": db_stats["total_columns"],
                        "status": "✅ Connected"
                    }
                else:
                    info["database"] = {
                        "status": f"❌ Error: {db_stats.get('error', 'Unknown error')}"
                    }
            except Exception as e:
                info["database"] = {
                    "status": f"❌ Error: {str(e)}"
                }
        
        return info
    
    def show_system_status(self):
        """Display system status information."""
        print("\n🚀 Intelligent Vehicle Registration AI System")
        print("=" * 60)
        
        info = self.get_system_info()
        
        print(f"🧠 System Type: {info['system_type']}")
        print(f"📊 Database: {info.get('database', {}).get('status', '❌ Unknown')}")
        if 'database' in info and 'total_records' in info['database']:
            print(f"   Records: {info['database']['total_records']:,}")
            print(f"   Columns: {info['database']['total_columns']}")
        
        print(f"🤖 AI Features: {'✅ Enabled' if info['has_openai_key'] else '❌ Disabled (No API Key)'}")
        
        print(f"\n🎯 Capabilities:")
        for capability in info['capabilities']:
            print(f"   ✅ {capability}")
        
        if not info['has_openai_key']:
            print(f"\n💡 To enable AI features, set your OpenAI API key:")
            print(f"   export OPENAI_API_KEY='your-api-key-here'")
    
    def run_interactive_mode(self):
        """Run the application in interactive mode."""
        if not self.initialized:
            print("❌ System not initialized. Please initialize first.")
            return
        
        print(f"\n💬 Interactive Mode - Ask ANY question!")
        print(f"🤖 I'll intelligently interpret and answer your questions about Telangana vehicle data")
        print(f"Type 'help' for suggestions, 'quit' to exit")
        print("=" * 70)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() in ['help', 'h']:
                    print(f"\n💡 Suggested questions:")
                    for i, suggestion in enumerate(self.get_suggested_questions()[:10], 1):
                        print(f"   {i}. {suggestion}")
                    print(f"\n💡 You can also ask ANY question - I'll interpret it intelligently!")
                    continue
                
                if not question:
                    continue
                
                # Process the question
                result = self.process_question(question)
                
                if result['success']:
                    print(f"\n✅ Answer:")
                    print(f"{result['answer']}")
                    
                    # Show if question was rewritten
                    if result.get('rewritten_question') and result['rewritten_question'] != question:
                        print(f"\n🔄 I interpreted your question as: {result['rewritten_question']}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")


def main():
    """Main function to run the application."""
    print("🚀 Starting Intelligent Vehicle Registration AI System")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OpenAI API key found. AI features will be limited.")
        print("   Set OPENAI_API_KEY environment variable for full functionality.")
    
    try:
        # Initialize the app
        app = IntelligentVehicleRegistrationApp()
        app.initialize()
        
        # Run interactive mode
        app.run_interactive_mode()
        
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print(f"   Please check your database and configuration.")


if __name__ == "__main__":
    main()