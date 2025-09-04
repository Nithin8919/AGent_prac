"""
Intelligent Citizen Agent - Natural Language Question Parser
==========================================================

This agent uses LLM to intelligently interpret any question and convert it
into relevant queries about the Telangana vehicle registration database.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llamaindex_connector import VehicleRegistrationConnector

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be asked."""
    COUNT = "count"
    LIST = "list"
    COMPARISON = "comparison"
    TREND = "trend"
    DETAIL = "detail"
    ANALYSIS = "analysis"


@dataclass
class QueryIntent:
    """Structured representation of a citizen's query intent."""
    original_question: str
    rewritten_question: str
    query_type: QueryType
    entities: Dict[str, Any]
    confidence: float
    clarification_needed: bool = False
    clarification_message: Optional[str] = None


class IntelligentCitizenAgent:
    """Intelligent agent for processing any natural language question."""
    
    def __init__(self):
        """Initialize the intelligent citizen agent."""
        self.connector = VehicleRegistrationConnector(db_path=Path("../db/vehicles.db"))
        self.connector.initialize()
        logger.info("Intelligent Citizen Agent initialized with LLM capabilities")
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process any question and provide a comprehensive answer."""
        try:
            # Step 1: Validate and interpret the question
            is_valid, error = self.validate_question(question)
            if not is_valid:
                return {
                    'success': False,
                    'error': error,
                    'answer': None
                }
            
            # Step 2: Intelligently rewrite and interpret the question
            intent = self.interpret_question(question)
            
            # Step 3: Generate comprehensive answer
            answer = self.generate_comprehensive_answer(intent)
            
            return {
                'success': True,
                'answer': answer,
                'intent': intent,
                'original_question': question,
                'rewritten_question': intent.rewritten_question
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'success': False,
                'error': f"Sorry, I encountered an error processing your question: {str(e)}",
                'answer': None
            }
    
    def validate_question(self, question: str) -> Tuple[bool, Optional[str]]:
        """Validate if a question is appropriate for the system."""
        if not question or len(question.strip()) < 3:
            return False, "Please ask a more specific question."
        
        # Accept any question - let the LLM handle interpretation
        return True, None
    
    def interpret_question(self, question: str) -> QueryIntent:
        """Use LLM to intelligently interpret and rewrite any question."""
        try:
            # Use the connector's LLM to interpret the question
            prompt = f"""
            You are an intelligent assistant for a Telangana vehicle registration database with 1.88 million records.
            
            User's question: "{question}"
            
            Your task:
            1. If this question is NOT related to vehicles, transport, or Telangana data, 
               intelligently rewrite it to be relevant to the vehicle registration database.
            2. If it IS related, keep it as is but make it more specific and actionable.
            3. Classify the rewritten question into one of these categories:
               - COUNT: Questions asking for counts, totals, numbers
               - LIST: Questions asking for lists, rankings, top items  
               - COMPARISON: Questions comparing different categories
               - TREND: Questions about trends over time
               - DETAIL: Questions asking for specific details
               - ANALYSIS: Questions requiring complex analysis
            
            Available data includes: vehicle registrations, manufacturers (Honda, Hero, Bajaj, Maruti, etc.), 
            fuel types (PETROL, DIESEL, BATTERY, etc.), body types, RTO offices, registration dates, 
            engine specifications, etc.
            
            Respond in this exact format:
            REWRITTEN_QUESTION: [your rewritten question]
            CATEGORY: [COUNT/LIST/COMPARISON/TREND/DETAIL/ANALYSIS]
            ENTITIES: [key entities mentioned, comma-separated]
            """
            
            response = self.connector.query_engine.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse the response
            lines = response_text.split('\n')
            rewritten_question = question  # fallback
            category = 'COUNT'  # fallback
            entities = []
            
            for line in lines:
                if line.startswith('REWRITTEN_QUESTION:'):
                    rewritten_question = line.replace('REWRITTEN_QUESTION:', '').strip()
                elif line.startswith('CATEGORY:'):
                    category = line.replace('CATEGORY:', '').strip().upper()
                elif line.startswith('ENTITIES:'):
                    entities_text = line.replace('ENTITIES:', '').strip()
                    entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            
            # Map to QueryType
            type_mapping = {
                'COUNT': QueryType.COUNT,
                'LIST': QueryType.LIST,
                'COMPARISON': QueryType.COMPARISON,
                'TREND': QueryType.TREND,
                'DETAIL': QueryType.DETAIL,
                'ANALYSIS': QueryType.ANALYSIS
            }
            
            query_type = type_mapping.get(category, QueryType.COUNT)
            
            return QueryIntent(
                original_question=question,
                rewritten_question=rewritten_question,
                query_type=query_type,
                entities={'mentioned': entities},
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Error interpreting question: {e}")
            # Fallback to basic parsing
            return QueryIntent(
                original_question=question,
                rewritten_question=question,
                query_type=QueryType.COUNT,
                entities={},
                confidence=0.3
            )
    
    def generate_comprehensive_answer(self, intent: QueryIntent) -> str:
        """Generate a comprehensive answer using the database."""
        try:
            # Use the rewritten question for better results
            query_result = self.connector.query(intent.rewritten_question)
            
            if not query_result['success']:
                # If the rewritten question fails, try the original
                query_result = self.connector.query(intent.original_question)
            
            if query_result['success']:
                # Enhance the answer with additional context
                enhanced_answer = self.enhance_answer(
                    query_result['answer'], 
                    intent, 
                    query_result.get('sql_query', '')
                )
                return enhanced_answer
            else:
                # Provide a helpful response even if query fails
                return self.provide_helpful_response(intent, query_result.get('error', ''))
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while processing your question. Please try rephrasing it or ask about vehicle registrations, manufacturers, fuel types, or other transport-related data."
    
    def enhance_answer(self, base_answer: str, intent: QueryIntent, sql_query: str) -> str:
        """Enhance the answer with additional context and insights."""
        try:
            # Use LLM to enhance the answer
            prompt = f"""
            You are providing information about Telangana vehicle registration data to a citizen.
            
            Original question: "{intent.original_question}"
            Rewritten question: "{intent.rewritten_question}"
            Query type: {intent.query_type.value}
            Base answer: "{base_answer}"
            
            Please enhance this answer to be:
            1. More comprehensive and informative
            2. Citizen-friendly and easy to understand
            3. Include relevant context about Telangana transport
            4. Add insights or comparisons where helpful
            5. Keep it concise but thorough
            
            Provide a well-structured, informative response.
            """
            
            response = self.connector.query_engine.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing answer: {e}")
            return base_answer
    
    def provide_helpful_response(self, intent: QueryIntent, error: str) -> str:
        """Provide a helpful response when the query fails."""
        try:
            prompt = f"""
            The user asked: "{intent.original_question}"
            The system couldn't find specific data for this query.
            
            Please provide a helpful response that:
            1. Acknowledges their question
            2. Explains what data is available in the Telangana vehicle registration database
            3. Suggests related questions they could ask
            4. Is friendly and helpful
            
            Available data includes: vehicle registrations, manufacturers, fuel types, body types, RTO offices, etc.
            """
            
            response = self.connector.query_engine.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error providing helpful response: {e}")
            return f"I understand you're asking about '{intent.original_question}'. While I couldn't find specific data for that query, I can help you with information about Telangana vehicle registrations, including manufacturers, fuel types, vehicle types, and RTO offices. Could you try rephrasing your question?"
    
    def get_suggested_questions(self) -> List[str]:
        """Get a list of suggested questions citizens can ask."""
        return [
            "How many vehicles are registered in Telangana?",
            "What are the top 5 vehicle manufacturers?",
            "How many vehicles use petrol vs diesel?",
            "Which RTO office has the most registrations?",
            "How many electric vehicles are registered?",
            "What is the distribution of vehicle types?",
            "Which manufacturer has the highest market share?",
            "What are the most popular vehicle models?",
            "How many vehicles were registered in Hyderabad?",
            "What is the trend in vehicle registrations over time?",
            "How many motorcycles vs cars are registered?",
            "What are the fuel type preferences in Telangana?",
            "Which areas have the most vehicle registrations?",
            "How many vehicles have CNG fuel?",
            "What is the average engine capacity of registered vehicles?"
        ]


def main():
    """Test the intelligent citizen agent."""
    print("🧑‍💼 Testing Intelligent Citizen Agent")
    print("=" * 50)
    
    agent = IntelligentCitizenAgent()
    
    test_questions = [
        "How many vehicles are registered?",
        "What are the top 5 manufacturers?",
        "How many vehicles use petrol vs diesel?",
        "Which RTO office has the most registrations?",
        "How many vehicles were registered in 2024?",
        "Tell me about Telangana transport",
        "What's the weather like?",  # Non-related question
        "How many questions are there in Telangana?"  # Ambiguous question
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = agent.process_question(question)
        
        if result['success']:
            print(f"✅ Answer: {result['answer']}")
            if result.get('rewritten_question') != question:
                print(f"🔄 Rewritten as: {result['rewritten_question']}")
        else:
            print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()