# src/agents/langgraph_integration.py
"""
LangGraph Integration for Existing Agents
=========================================

This module adapts your existing LlamaIndex agents to work with LangGraph orchestration
while preserving all their functionality.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

# LangGraph imports  
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import existing agents
sys.path.append(str(Path(__file__).parent))
from citizen_agent import IntelligentCitizenAgent, CitizenAgent, QueryIntent, QueryType
from data_agent import DataAgent
from analytics_agent import AnalyticsAgent
from llamaindex_connector import VehicleRegistrationConnector

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED STATE FOR LANGGRAPH
# =============================================================================

class IntegratedAgentState(TypedDict):
    """State that works with existing agents"""
    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Query processing
    original_query: str
    query_intent: Optional[Dict[str, Any]]  # From CitizenAgent
    
    # Data results
    data_result: Optional[Dict[str, Any]]   # From DataAgent
    analytics_result: Optional[Dict[str, Any]]  # From AnalyticsAgent
    
    # Workflow management
    current_agent: str
    next_agent: str
    completed_agents: List[str]
    
    # Final output
    final_answer: str
    confidence_score: float
    
    # Memory and context
    conversation_history: List[Dict[str, Any]]
    error_messages: List[str]


# =============================================================================
# LANGGRAPH WRAPPERS FOR EXISTING AGENTS
# =============================================================================

class LangGraphCitizenAgent:
    """LangGraph wrapper for existing CitizenAgent/IntelligentCitizenAgent"""
    
    def __init__(self, db_path: str = "../db/vehicles.db"):
        """Initialize with existing intelligent citizen agent"""
        try:
            # Try intelligent agent first
            self.agent = IntelligentCitizenAgent()
            self.agent_type = "intelligent"
            logger.info("✅ Using IntelligentCitizenAgent")
        except Exception as e:
            # Fallback to basic agent
            logger.warning(f"IntelligentCitizenAgent failed: {e}, using basic CitizenAgent")
            self.agent = CitizenAgent()
            self.agent_type = "basic"
        
        self.name = "citizen_agent"
    
    def __call__(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """Process query using existing citizen agent"""
        
        try:
            # Get the user's query
            last_message = state["messages"][-1] if state["messages"] else None
            if not last_message:
                state["next_agent"] = "end"
                return state
            
            query = last_message.content
            state["original_query"] = query
            
            # Process with appropriate agent
            if self.agent_type == "intelligent":
                result = self.agent.process_question(query)
                
                if result['success']:
                    state["query_intent"] = {
                        "intent": result.get('intent'),
                        "original": result['original_question'],
                        "rewritten": result['rewritten_question']
                    }
                    
                    # Add processing message
                    state["messages"].append(AIMessage(
                        content=f"🧑‍💼 **Intelligent Query Processing**\n\nOriginal: {query}\nRewritten: {result['rewritten_question']}\n\n🔄 Routing to Data Agent..."
                    ))
                    
                    state["next_agent"] = "data_agent"
                else:
                    state["error_messages"].append(result.get('error', 'Query processing failed'))
                    state["next_agent"] = "end"
            
            else:
                # Basic agent processing
                is_valid, error = self.agent.validate_question(query)
                if not is_valid:
                    state["error_messages"].append(error)
                    state["next_agent"] = "end"
                    return state
                
                intent = self.agent.parse_question(query)
                state["query_intent"] = {
                    "query_type": intent.query_type.value,
                    "entity": intent.entity,
                    "filters": intent.filters,
                    "limit": intent.limit
                }
                
                state["messages"].append(AIMessage(
                    content=f"🧑‍💼 **Query Parsed**\n\nType: {intent.query_type.value}\nEntity: {intent.entity}\n\n🔄 Fetching data..."
                ))
                
                state["next_agent"] = "data_agent"
            
            state["completed_agents"].append("citizen_agent")
            return state
            
        except Exception as e:
            logger.error(f"LangGraphCitizenAgent failed: {e}")
            state["error_messages"].append(f"Citizen agent error: {str(e)}")
            state["next_agent"] = "end"
            return state


class LangGraphDataAgent:
    """LangGraph wrapper for existing DataAgent"""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: str = None):
        """Initialize with existing data agent"""
        self.agent = DataAgent(db_path, openai_api_key)
        self.name = "data_agent"
        
        # Initialize the agent
        try:
            self.agent.initialize()
            logger.info("✅ DataAgent initialized")
        except Exception as e:
            logger.warning(f"DataAgent initialization warning: {e}")
    
    def __call__(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """Execute data query using existing agent"""
        
        try:
            query_intent = state.get("query_intent", {})
            original_query = state.get("original_query", "")
            
            if not query_intent and not original_query:
                state["error_messages"].append("No query to process")
                state["next_agent"] = "end"
                return state
            
            # Convert to QueryIntent if needed (for basic agent compatibility)
            if "query_type" in query_intent:
                # Basic agent format
                from citizen_agent import QueryIntent, QueryType
                intent_obj = QueryIntent(
                    query_type=QueryType(query_intent["query_type"]),
                    entity=query_intent.get("entity", ""),
                    filters=query_intent.get("filters", {}),
                    limit=query_intent.get("limit")
                )
                result = self.agent.execute_query(intent_obj)
            else:
                # Intelligent agent - just query directly
                result = self.agent.connector.query(original_query)
            
            state["data_result"] = result
            
            if result.get("success", False):
                answer_preview = result.get("answer", "")[:300]
                state["messages"].append(AIMessage(
                    content=f"📊 **Data Retrieved Successfully**\n\n{answer_preview}...\n\n🔄 Analyzing data..."
                ))
                
                # Determine if analytics needed
                query_type = query_intent.get("query_type", "general")
                if query_type in ["comparison", "trend", "distribution"]:
                    state["next_agent"] = "analytics_agent"
                else:
                    state["next_agent"] = "synthesis_agent"  # Skip analytics for simple queries
            else:
                error_msg = result.get("error", "Data retrieval failed")
                state["error_messages"].append(error_msg)
                state["messages"].append(AIMessage(
                    content=f"❌ **Data Retrieval Failed**\n\n{error_msg}"
                ))
                state["next_agent"] = "end"
            
            state["completed_agents"].append("data_agent")
            return state
            
        except Exception as e:
            logger.error(f"LangGraphDataAgent failed: {e}")
            state["error_messages"].append(f"Data agent error: {str(e)}")
            state["next_agent"] = "end"
            return state


class LangGraphAnalyticsAgent:
    """LangGraph wrapper for existing AnalyticsAgent"""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: str = None):
        """Initialize with existing analytics agent"""
        self.agent = AnalyticsAgent(db_path, openai_api_key)
        self.name = "analytics_agent"
        
        # Test connection
        if self.agent.test_connection():
            logger.info("✅ AnalyticsAgent connected")
        else:
            logger.warning("⚠️ AnalyticsAgent connection issues")
    
    def __call__(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """Perform analytics using existing agent"""
        
        try:
            data_result = state.get("data_result", {})
            query_intent = state.get("query_intent", {})
            
            if not data_result or not data_result.get("success", False):
                # Skip analytics if no data
                state["next_agent"] = "synthesis_agent"
                state["completed_agents"].append("analytics_agent")
                return state
            
            # Perform analytics based on query type
            analytics_results = []
            query_type = query_intent.get("query_type", "general")
            
            try:
                if query_type == "comparison":
                    entities = ["manufacturer", "fuel_type"]  # Example entities
                    for entity in entities:
                        result = self.agent.compare_entities(entity, ["honda", "hero", "bajaj"])
                        if result.insights:
                            analytics_results.extend(result.insights)
                
                elif query_type == "trend":
                    result = self.agent.analyze_trends("total", [2022, 2023, 2024, 2025])
                    if result.insights:
                        analytics_results.extend(result.insights)
                
                else:
                    # Market share analysis
                    result = self.agent.calculate_market_share("manufacturer")
                    if result and len(result) > 0:
                        analytics_results.append("Market share analysis completed")
            
            except Exception as analytics_error:
                logger.warning(f"Analytics processing warning: {analytics_error}")
                analytics_results = ["Basic analytics completed with limited insights"]
            
            state["analytics_result"] = {
                "insights": analytics_results,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat()
            }
            
            if analytics_results:
                insights_text = "\n• ".join(analytics_results)
                state["messages"].append(AIMessage(
                    content=f"📈 **Analytics Complete**\n\n• {insights_text}\n\n🔄 Synthesizing final answer..."
                ))
            else:
                state["messages"].append(AIMessage(
                    content="📈 **Analytics Complete**\n\nBasic analysis performed\n\n🔄 Synthesizing final answer..."
                ))
            
            state["next_agent"] = "synthesis_agent"
            state["completed_agents"].append("analytics_agent")
            return state
            
        except Exception as e:
            logger.error(f"LangGraphAnalyticsAgent failed: {e}")
            # Continue to synthesis even if analytics fails
            state["analytics_result"] = {"error": str(e)}
            state["next_agent"] = "synthesis_agent"
            state["completed_agents"].append("analytics_agent")
            return state


class LangGraphSynthesisAgent:
    """Synthesis agent that combines all results"""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize synthesis agent with LLM"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=self.openai_api_key
            )
        else:
            self.llm = None
            logger.warning("No OpenAI key - synthesis will be basic")
        
        self.name = "synthesis_agent"
    
    def __call__(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """Create comprehensive final answer"""
        
        try:
            original_query = state.get("original_query", "")
            data_result = state.get("data_result", {})
            analytics_result = state.get("analytics_result", {})
            
            # Build final answer
            if self.llm and data_result.get("success"):
                # AI-enhanced synthesis
                final_answer = self._create_ai_synthesis(original_query, data_result, analytics_result)
            else:
                # Basic synthesis
                final_answer = self._create_basic_synthesis(original_query, data_result, analytics_result)
            
            state["final_answer"] = final_answer
            state["confidence_score"] = self._calculate_confidence(data_result, analytics_result)
            
            # Add to conversation history
            state["conversation_history"].append({
                "query": original_query,
                "answer": final_answer,
                "timestamp": datetime.now().isoformat(),
                "agents_used": state["completed_agents"]
            })
            
            state["messages"].append(AIMessage(
                content=f"✅ **Analysis Complete**\n\n{final_answer}"
            ))
            
            state["next_agent"] = "end"
            state["completed_agents"].append("synthesis_agent")
            
            return state
            
        except Exception as e:
            logger.error(f"LangGraphSynthesisAgent failed: {e}")
            state["final_answer"] = f"Analysis completed with errors: {str(e)}"
            state["next_agent"] = "end"
            return state
    
    def _create_ai_synthesis(self, query: str, data_result: Dict, analytics_result: Dict) -> str:
        """Create AI-enhanced synthesis"""
        
        try:
            synthesis_prompt = f"""
            Create a comprehensive answer for this vehicle registration query:
            
            Query: {query}
            
            Data Result: {json.dumps(data_result, indent=2)[:1000]}
            
            Analytics: {json.dumps(analytics_result, indent=2)[:500]}
            
            Provide a clear, factual answer that:
            1. Directly answers the question
            2. Includes specific numbers/data points
            3. Adds relevant insights
            4. Is suitable for policy makers
            
            Keep it concise but informative.
            """
            
            response = self.llm.invoke([SystemMessage(content=synthesis_prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"AI synthesis failed: {e}")
            return self._create_basic_synthesis(query, data_result, analytics_result)
    
    def _create_basic_synthesis(self, query: str, data_result: Dict, analytics_result: Dict) -> str:
        """Create basic synthesis without AI"""
        
        answer_parts = [f"**Answer to:** {query}\n"]
        
        # Add data result
        if data_result.get("success"):
            data_answer = data_result.get("answer", "Data retrieved successfully")
            answer_parts.append(f"**Data Analysis:**\n{data_answer}")
        else:
            answer_parts.append("**Data Analysis:** No specific data could be retrieved.")
        
        # Add analytics insights
        if analytics_result and analytics_result.get("insights"):
            insights = analytics_result["insights"]
            answer_parts.append(f"\n**Key Insights:**")
            for insight in insights[:5]:  # Limit to top 5
                answer_parts.append(f"• {insight}")
        
        return "\n".join(answer_parts)
    
    def _calculate_confidence(self, data_result: Dict, analytics_result: Dict) -> float:
        """Calculate confidence score"""
        
        confidence = 0.5  # Base confidence
        
        if data_result.get("success"):
            confidence += 0.3
        
        if analytics_result and analytics_result.get("insights"):
            confidence += 0.2
        
        return min(confidence, 1.0)


# =============================================================================
# INTEGRATED ORCHESTRATOR
# =============================================================================

class IntegratedVehicleRegistrationSystem:
    """Orchestrator that uses existing agents with LangGraph"""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: str = None):
        """Initialize integrated system"""
        
        self.db_path = db_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize memory
        self.memory = MemorySaver()
        
        # Initialize integrated agents (wrappers for existing agents)
        self.agents = {
            "citizen_agent": LangGraphCitizenAgent(db_path),
            "data_agent": LangGraphDataAgent(db_path, self.openai_api_key),
            "analytics_agent": LangGraphAnalyticsAgent(db_path, self.openai_api_key),
            "synthesis_agent": LangGraphSynthesisAgent(self.openai_api_key)
        }
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("✅ IntegratedVehicleRegistrationSystem initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow using existing agents"""
        
        workflow = StateGraph(IntegratedAgentState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, agent)
        
        # Routing logic
        def route_next(state: IntegratedAgentState) -> str:
            next_agent = state.get("next_agent", "end")
            if next_agent == "end":
                return END
            return next_agent
        
        # Connect agents
        workflow.add_conditional_edges("citizen_agent", route_next)
        workflow.add_conditional_edges("data_agent", route_next)  
        workflow.add_conditional_edges("analytics_agent", route_next)
        workflow.add_conditional_edges("synthesis_agent", route_next)
        
        # Entry point
        workflow.add_edge(START, "citizen_agent")
        
        return workflow
    
    def process_query(self, query: str, thread_id: str = "main") -> Dict[str, Any]:
        """Process query through integrated workflow"""
        
        try:
            # Initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_intent": None,
                "data_result": None,
                "analytics_result": None,
                "current_agent": "citizen_agent",
                "next_agent": "citizen_agent",
                "completed_agents": [],
                "final_answer": "",
                "confidence_score": 0.0,
                "conversation_history": [],
                "error_messages": []
            }
            
            # Configure memory
            config = {"configurable": {"thread_id": thread_id}}
            
            # Run workflow
            result = self.app.invoke(initial_state, config=config)
            
            return {
                "success": True,
                "final_answer": result.get("final_answer", ""),
                "messages": [msg.content for msg in result.get("messages", [])],
                "agents_used": result.get("completed_agents", []),
                "confidence_score": result.get("confidence_score", 0.0),
                "data_result": result.get("data_result", {}),
                "analytics_result": result.get("analytics_result", {}),
                "errors": result.get("error_messages", [])
            }
            
        except Exception as e:
            logger.error(f"Integrated system query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_answer": f"I encountered an error: {str(e)}"
            }


# =============================================================================
# UPDATED MAIN APPLICATION
# =============================================================================

class IntegratedApp:
    """Updated main application using integrated system"""
    
    def __init__(self):
        self.system = None
    
    def initialize(self, db_path: str = "../db/vehicles.db") -> bool:
        """Initialize integrated system"""
        
        try:
            print("🚀 Initializing Integrated Vehicle Registration System...")
            print("   LlamaIndex agents + LangGraph orchestration")
            
            self.system = IntegratedVehicleRegistrationSystem(db_path)
            
            print("✅ System ready!")
            print(f"   Database: {db_path}")
            print(f"   Agents: {len(self.system.agents)} integrated")
            print(f"   Memory: Persistent across conversations")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    def run_interactive(self):
        """Run interactive mode"""
        
        if not self.system:
            print("❌ System not initialized!")
            return
        
        print("""
🧠 **Integrated Vehicle Registration AI**
Your existing LlamaIndex agents now work with LangGraph orchestration!

🤖 **Active Agents:**
• CitizenAgent (Intelligent or Basic) - Query understanding
• DataAgent - LlamaIndex data retrieval  
• AnalyticsAgent - Advanced analytics & insights
• SynthesisAgent - AI-powered final answers

Ask any question about vehicle registration data!
Type 'quit' to exit.
        """)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                if not query:
                    continue
                
                print("🔄 Processing through integrated multi-agent workflow...")
                
                result = self.system.process_query(query)
                
                if result["success"]:
                    print(f"\n✅ **Final Answer:**")
                    print(f"{result['final_answer']}")
                    
                    print(f"\n📊 **System Performance:**")
                    print(f"   Agents used: {', '.join(result['agents_used'])}")
                    print(f"   Confidence: {result['confidence_score']:.1%}")
                    
                    if result.get("errors"):
                        print(f"   Warnings: {len(result['errors'])} minor issues")
                    
                else:
                    print(f"\n❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point for integrated system"""
    
    app = IntegratedApp()
    
    # Check database
    db_path = Path("../db/vehicles.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("Please run data ingestion first: python src/data_ingestion.py")
        return
    
    if app.initialize(str(db_path)):
        app.run_interactive()
    else:
        print("Failed to initialize. Please check your setup.")


if __name__ == "__main__":
    main()