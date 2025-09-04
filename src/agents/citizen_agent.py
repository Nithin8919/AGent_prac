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
        self.connector = VehicleRegistrationConnector(db_path=Path("db/vehicles.db"))
        self.connector.initialize()
        # Store LLM reference for direct access
        from llama_index.core import Settings
        self.llm = Settings.llm
        logger.info("Intelligent Citizen Agent initialized with LLM capabilities")
    
    def process_question(self, question: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Process any question and provide a comprehensive answer with context awareness."""
        try:
            # Step 1: Validate and interpret the question
            is_valid, error = self.validate_question(question)
            if not is_valid:
                return {
                    'success': False,
                    'error': error,
                    'answer': None
                }
            
            # Step 2: Check if this is a follow-up question that needs context
            if self.is_followup_question(question, conversation_history):
                return self.handle_followup_question(question, conversation_history)
            
            # Step 3: Check if this is an analytical question that needs LLM reasoning
            if self.is_analytical_question(question):
                return self.handle_analytical_question(question, conversation_history)
            
            # Step 4: Intelligently rewrite and interpret the question
            intent = self.interpret_question(question)
            
            # Step 4: Generate comprehensive answer
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
    
    def is_followup_question(self, question: str, conversation_history: List[str] = None) -> bool:
        """Check if this is a follow-up question that needs context."""
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        # Keywords that indicate follow-up questions
        followup_indicators = [
            "according to that", "based on that", "from above", "from the previous",
            "what changes", "how can we", "what should we", "what do you think",
            "what are your thoughts", "what recommendations", "what suggestions",
            "what can be done", "what improvements", "what next", "what else"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in followup_indicators)
    
    def is_analytical_question(self, question: str) -> bool:
        """Check if this is an analytical question that needs LLM reasoning, not SQL."""
        question_lower = question.lower()
        
        # Keywords that indicate analytical questions (not data queries)
        analytical_indicators = [
            "pollution levels", "environmental impact", "air quality", "emissions",
            "what are the implications", "what does this mean", "what are the effects",
            "how does this affect", "what are the consequences", "what are the benefits",
            "what are the risks", "what are the challenges", "what are the opportunities",
            "what are the trends", "what are the patterns", "what are the insights",
            "what are the recommendations", "what should be done", "what can be improved",
            "what are the solutions", "what are the alternatives", "what are the options",
            "what are the pros and cons", "what are the advantages", "what are the disadvantages",
            "what is the impact", "what is the significance", "what is the importance",
            "what is the relationship", "what is the correlation", "what is the connection",
            "what is the cause", "what is the reason", "what is the explanation",
            "what is the analysis", "what is the interpretation", "what is the conclusion"
        ]
        
        return any(indicator in question_lower for indicator in analytical_indicators)
    
    def handle_analytical_question(self, question: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Handle analytical questions using LLM reasoning with available data context."""
        try:
            # Get conversation context if available
            context_text = ""
            if conversation_history:
                recent_context = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                context_text = "\n".join([f"- {msg}" for msg in recent_context])
            
            # Create a comprehensive prompt for analytical questions
            prompt = f"""
            You are an AI assistant specializing in Telangana vehicle registration data analysis.
            
            Available data context:
            {context_text}
            
            Available database information:
            - Vehicle registration data with 1,880,183 records
            - Fields: registration_number, maker_name, model_description, body_type, 
              engine_cc, fuel_type, horsepower, seat_capacity, office_code, registration dates
            - Fuel types: PETROL, DIESEL, BATTERY, CNG, etc.
            - Vehicle types: motorcycles, cars, buses, trucks, etc.
            - Manufacturers: Honda, Hero, Bajaj, Maruti, etc.
            
            Question: "{question}"
            
            Please provide a comprehensive analytical response that:
            1. Acknowledges the limitations of the available data
            2. Provides insights based on what can be inferred from vehicle registration data
            3. Explains the relationship between vehicle data and the question asked
            4. Offers educated analysis and recommendations
            5. Suggests what additional data would be needed for more precise answers
            
            Be specific about Telangana context and provide actionable insights.
            Do NOT attempt to generate SQL queries for this type of analytical question.
            """
            
            response = self.llm.complete(prompt)
            answer = response.text.strip()
            
            return {
                'success': True,
                'answer': answer,
                'intent': QueryIntent(
                    original_question=question,
                    rewritten_question=question,
                    query_type=QueryType.ANALYSIS,
                    entities={'analytical': True},
                    confidence=0.9
                ),
                'original_question': question,
                'rewritten_question': question,
                'is_analytical': True
            }
            
        except Exception as e:
            logger.error(f"Error handling analytical question: {e}")
            return {
                'success': False,
                'error': f"Sorry, I had trouble processing your analytical question: {str(e)}",
                'answer': None
            }
    
    def handle_followup_question(self, question: str, conversation_history: List[str]) -> Dict[str, Any]:
        """Handle follow-up questions using conversation context."""
        try:
            # Get the last few messages for context
            recent_context = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            # Create a context-aware prompt
            context_text = "\n".join([f"- {msg}" for msg in recent_context])
            
            prompt = f"""
            You are an AI assistant helping with Telangana vehicle registration data analysis.
            
            Previous conversation context:
            {context_text}
            
            Current follow-up question: "{question}"
            
            Based on the previous context, provide a thoughtful, analytical response that:
            1. References the previous data/answers appropriately
            2. Provides insights, recommendations, or analysis
            3. Is specific to the Telangana vehicle registration context
            4. Offers actionable suggestions or observations
            
            Do NOT generate SQL queries for this type of question. Provide analytical insights instead.
            """
            
            response = self.llm.complete(prompt)
            answer = response.text.strip()
            
            return {
                'success': True,
                'answer': answer,
                'intent': QueryIntent(
                    original_question=question,
                    rewritten_question=question,
                    query_type=QueryType.ANALYSIS,
                    entities={'context_aware': True},
                    confidence=0.9
                ),
                'original_question': question,
                'rewritten_question': question,
                'is_followup': True
            }
            
        except Exception as e:
            logger.error(f"Error handling follow-up question: {e}")
            return {
                'success': False,
                'error': f"Sorry, I had trouble processing your follow-up question: {str(e)}",
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
            
            response = self.llm.complete(prompt)
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
            
            response = self.llm.complete(prompt)
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
            
            response = self.llm.complete(prompt)
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