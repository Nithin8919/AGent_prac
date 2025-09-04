# src/unified_system.py
"""
Unified Vehicle Registration AI System
=====================================

Integrates LlamaIndex (data processing) + LangGraph (orchestration) + Long-term Memory
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import json
from datetime import datetime
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import our existing LlamaIndex connector
sys.path.append(str(Path(__file__).parent))
from llamaindex_connector import VehicleRegistrationConnector

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED STATE DEFINITION
# =============================================================================

class UnifiedAgentState(TypedDict):
    """Unified state shared across all agents"""
    # Messages
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Query context
    original_query: str
    query_type: str
    entities: Dict[str, Any]
    
    # Data processing
    raw_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    
    # Agent coordination
    current_agent: str
    next_agent: str
    completed_agents: List[str]
    
    # Results
    analysis_results: Dict[str, Any]
    insights: List[str]
    final_answer: str
    
    # Memory
    conversation_memory: List[Dict[str, Any]]
    long_term_insights: Dict[str, Any]


# =============================================================================
# UNIFIED LLAMAINDEX CONNECTOR WITH MEMORY
# =============================================================================

class UnifiedLlamaIndexConnector(VehicleRegistrationConnector):
    """Enhanced LlamaIndex connector with memory capabilities"""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: str = None):
        super().__init__(db_path, openai_api_key)
        self.memory_store = {}
        self.query_history = []
        
    def query_with_memory(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query with context and memory"""
        
        # Add context to the question if available
        if context:
            enhanced_question = f"""
            Context: {json.dumps(context, indent=2)}
            
            Question: {question}
            
            Please answer considering the context above.
            """
        else:
            enhanced_question = question
        
        # Execute query
        result = self.query(enhanced_question)
        
        # Store in memory
        self.query_history.append({
            "question": question,
            "context": context,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_memory_context(self, query_type: str = None) -> Dict[str, Any]:
        """Get relevant memory context for current query"""
        
        if not self.query_history:
            return {}
        
        # Get recent relevant queries
        recent_queries = self.query_history[-5:]  # Last 5 queries
        
        memory_context = {
            "recent_queries": [q["question"] for q in recent_queries],
            "recent_results": [q["result"]["answer"] if q["result"]["success"] else "Failed" for q in recent_queries],
            "query_patterns": self._analyze_query_patterns()
        }
        
        return memory_context
    
    def _analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in query history"""
        
        if not self.query_history:
            return {}
        
        patterns = {
            "most_asked_about": [],
            "common_entities": [],
            "frequent_topics": []
        }
        
        # Simple pattern analysis
        all_questions = [q["question"].lower() for q in self.query_history]
        
        # Count common words (basic implementation)
        word_counts = {}
        for question in all_questions:
            words = question.split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns["frequent_topics"] = [word for word, count in top_words]
        
        return patterns


# =============================================================================
# LANGGRAPH AGENTS (Using LlamaIndex for data)
# =============================================================================

class CitizenAgent:
    """Citizen Agent - Understands and parses natural language queries"""
    
    def __init__(self, connector: UnifiedLlamaIndexConnector, llm: ChatOpenAI):
        self.connector = connector
        self.llm = llm
        self.name = "CitizenAgent"
    
    def __call__(self, state: UnifiedAgentState) -> UnifiedAgentState:
        """Process citizen's question"""
        
        try:
            # Get the last message
            last_message = state["messages"][-1] if state["messages"] else None
            if not last_message:
                return state
            
            query = last_message.content
            state["original_query"] = query
            
            # Use LLM to analyze the query
            analysis_prompt = f"""
            Analyze this vehicle registration query and extract:
            1. Query type (count, comparison, trend, distribution, detail)
            2. Key entities (manufacturers, fuel types, locations, etc.)
            3. Intent and expected answer type
            
            Query: "{query}"
            
            Respond in JSON format:
            {{
                "query_type": "type",
                "entities": {{"key": "value"}},
                "intent": "description",
                "next_agent": "data_agent"
            }}
            """
            
            response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
            
            # Parse response (simplified)
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    state["query_type"] = analysis.get("query_type", "general")
                    state["entities"] = analysis.get("entities", {})
                    state["next_agent"] = analysis.get("next_agent", "data_agent")
                except json.JSONDecodeError:
                    state["query_type"] = "general"
                    state["entities"] = {}
                    state["next_agent"] = "data_agent"
            else:
                state["query_type"] = "general"
                state["entities"] = {}
                state["next_agent"] = "data_agent"
            
            # Add to completed agents
            state["completed_agents"].append("citizen_agent")
            
            # Add analysis message
            state["messages"].append(AIMessage(
                content=f"🧑‍💼 **Query Analysis Complete**\n\nType: {state['query_type']}\nEntities: {state['entities']}\n\n🔄 Routing to Data Agent..."
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"CitizenAgent failed: {e}")
            state["messages"].append(AIMessage(content=f"Error analyzing query: {str(e)}"))
            state["next_agent"] = "end"
            return state


class DataAgent:
    """Data Agent - Retrieves data using LlamaIndex"""
    
    def __init__(self, connector: UnifiedLlamaIndexConnector, llm: ChatOpenAI):
        self.connector = connector
        self.llm = llm
        self.name = "DataAgent"
    
    def __call__(self, state: UnifiedAgentState) -> UnifiedAgentState:
        """Retrieve relevant data"""
        
        try:
            query = state["original_query"]
            query_type = state.get("query_type", "general")
            entities = state.get("entities", {})
            
            # Get memory context
            memory_context = self.connector.get_memory_context(query_type)
            
            # Execute query with context
            result = self.connector.query_with_memory(query, {
                "query_type": query_type,
                "entities": entities,
                "memory": memory_context
            })
            
            # Store results
            state["raw_data"] = {
                "query_result": result,
                "memory_context": memory_context
            }
            
            # Determine next step based on query type
            if query_type in ["count", "detail"]:
                state["next_agent"] = "synthesis_agent"  # Skip analytics for simple queries
            else:
                state["next_agent"] = "analytics_agent"
            
            # Add to completed
            state["completed_agents"].append("data_agent")
            
            # Add data message
            if result["success"]:
                state["messages"].append(AIMessage(
                    content=f"📊 **Data Retrieved**\n\n{result['answer'][:500]}...\n\n🔄 Processing with analytics..."
                ))
            else:
                state["messages"].append(AIMessage(
                    content=f"❌ Data retrieval failed: {result.get('error', 'Unknown error')}"
                ))
                state["next_agent"] = "end"
            
            return state
            
        except Exception as e:
            logger.error(f"DataAgent failed: {e}")
            state["messages"].append(AIMessage(content=f"Error retrieving data: {str(e)}"))
            state["next_agent"] = "end"
            return state


class AnalyticsAgent:
    """Analytics Agent - Performs analysis on retrieved data"""
    
    def __init__(self, connector: UnifiedLlamaIndexConnector, llm: ChatOpenAI):
        self.connector = connector
        self.llm = llm
        self.name = "AnalyticsAgent"
    
    def __call__(self, state: UnifiedAgentState) -> UnifiedAgentState:
        """Perform analytics on the data"""
        
        try:
            raw_data = state.get("raw_data", {})
            query_type = state.get("query_type", "general")
            original_query = state["original_query"]
            
            if not raw_data or not raw_data.get("query_result", {}).get("success"):
                state["next_agent"] = "synthesis_agent"
                state["completed_agents"].append("analytics_agent")
                return state
            
            # Perform additional analysis based on query type
            if query_type == "comparison":
                analysis_result = self._perform_comparison_analysis(original_query, raw_data)
            elif query_type == "trend":
                analysis_result = self._perform_trend_analysis(original_query, raw_data)
            else:
                analysis_result = self._perform_general_analysis(original_query, raw_data)
            
            state["processed_data"] = analysis_result
            state["next_agent"] = "insight_agent"
            state["completed_agents"].append("analytics_agent")
            
            state["messages"].append(AIMessage(
                content=f"📈 **Analytics Complete**\n\nAnalysis type: {query_type}\nKey findings processed\n\n🔄 Generating insights..."
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"AnalyticsAgent failed: {e}")
            state["messages"].append(AIMessage(content=f"Error in analytics: {str(e)}"))
            state["next_agent"] = "synthesis_agent"
            return state
    
    def _perform_comparison_analysis(self, query: str, raw_data: Dict) -> Dict[str, Any]:
        """Perform comparison analysis"""
        return {
            "type": "comparison",
            "summary": "Comparison analysis performed",
            "data": raw_data["query_result"]
        }
    
    def _perform_trend_analysis(self, query: str, raw_data: Dict) -> Dict[str, Any]:
        """Perform trend analysis"""
        return {
            "type": "trend",
            "summary": "Trend analysis performed",
            "data": raw_data["query_result"]
        }
    
    def _perform_general_analysis(self, query: str, raw_data: Dict) -> Dict[str, Any]:
        """Perform general analysis"""
        return {
            "type": "general",
            "summary": "General analysis performed",
            "data": raw_data["query_result"]
        }


class InsightAgent:
    """Insight Agent - Generates AI-powered insights"""
    
    def __init__(self, connector: UnifiedLlamaIndexConnector, llm: ChatOpenAI):
        self.connector = connector
        self.llm = llm
        self.name = "InsightAgent"
    
    def __call__(self, state: UnifiedAgentState) -> UnifiedAgentState:
        """Generate insights using LLM"""
        
        try:
            original_query = state["original_query"]
            raw_data = state.get("raw_data", {})
            processed_data = state.get("processed_data", {})
            
            # Create insight prompt
            insight_prompt = f"""
            Generate comprehensive insights for this vehicle registration analysis:
            
            Original Question: {original_query}
            
            Data: {json.dumps(raw_data.get("query_result", {}), indent=2)[:1000]}
            
            Analysis: {json.dumps(processed_data, indent=2)[:500]}
            
            Provide:
            1. Key insights (3-5 bullet points)
            2. Implications for policy makers
            3. Recommendations
            4. Data reliability assessment
            
            Be specific, actionable, and data-driven.
            """
            
            response = self.llm.invoke([SystemMessage(content=insight_prompt)])
            
            insights = response.content
            state["insights"] = [insights]
            state["next_agent"] = "synthesis_agent"
            state["completed_agents"].append("insight_agent")
            
            state["messages"].append(AIMessage(
                content=f"💡 **Insights Generated**\n\n{insights}\n\n🔄 Preparing final synthesis..."
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"InsightAgent failed: {e}")
            state["messages"].append(AIMessage(content=f"Error generating insights: {str(e)}"))
            state["next_agent"] = "synthesis_agent"
            return state


class SynthesisAgent:
    """Synthesis Agent - Combines all results into final answer"""
    
    def __init__(self, connector: UnifiedLlamaIndexConnector, llm: ChatOpenAI):
        self.connector = connector
        self.llm = llm
        self.name = "SynthesisAgent"
    
    def __call__(self, state: UnifiedAgentState) -> UnifiedAgentState:
        """Create final comprehensive answer"""
        
        try:
            original_query = state["original_query"]
            raw_data = state.get("raw_data", {})
            insights = state.get("insights", [])
            
            # Create synthesis prompt
            synthesis_prompt = f"""
            Create a comprehensive final answer for:
            
            Question: {original_query}
            
            Raw Data: {json.dumps(raw_data.get("query_result", {}), indent=2)[:800]}
            
            Insights: {' '.join(insights)[:500]}
            
            Create a well-structured final answer that:
            1. Directly answers the question
            2. Provides specific numbers and facts
            3. Includes insights and implications
            4. Is clear and actionable
            
            Format as a comprehensive response suitable for policy makers.
            """
            
            response = self.llm.invoke([SystemMessage(content=synthesis_prompt)])
            
            final_answer = response.content
            state["final_answer"] = final_answer
            state["next_agent"] = "end"
            state["completed_agents"].append("synthesis_agent")
            
            # Update long-term memory
            state["long_term_insights"] = {
                "query": original_query,
                "answer": final_answer,
                "timestamp": datetime.now().isoformat(),
                "agents_used": state["completed_agents"]
            }
            
            state["messages"].append(AIMessage(
                content=f"✅ **Analysis Complete**\n\n{final_answer}"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"SynthesisAgent failed: {e}")
            state["messages"].append(AIMessage(content=f"Error in synthesis: {str(e)}"))
            state["next_agent"] = "end"
            return state


# =============================================================================
# UNIFIED ORCHESTRATOR
# =============================================================================

class UnifiedVehicleRegistrationSystem:
    """Unified system combining LlamaIndex + LangGraph + Memory"""
    
    def __init__(self, db_path: str = "../db/vehicles.db", openai_api_key: str = None):
        """Initialize the unified system"""
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        
        # Initialize enhanced connector
        self.connector = UnifiedLlamaIndexConnector(db_path, self.openai_api_key)
        self.connector.initialize()
        
        # Initialize memory
        self.memory = MemorySaver()
        
        # Initialize agents
        self.agents = {
            "citizen_agent": CitizenAgent(self.connector, self.llm),
            "data_agent": DataAgent(self.connector, self.llm),
            "analytics_agent": AnalyticsAgent(self.connector, self.llm),
            "insight_agent": InsightAgent(self.connector, self.llm),
            "synthesis_agent": SynthesisAgent(self.connector, self.llm)
        }
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("✅ Unified Vehicle Registration System initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the unified workflow"""
        
        workflow = StateGraph(UnifiedAgentState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, agent)
        
        # Routing function
        def route_next(state: UnifiedAgentState) -> str:
            next_agent = state.get("next_agent", "end")
            if next_agent == "end":
                return END
            return next_agent
        
        # Set up routing
        workflow.add_conditional_edges("citizen_agent", route_next)
        workflow.add_conditional_edges("data_agent", route_next)
        workflow.add_conditional_edges("analytics_agent", route_next)
        workflow.add_conditional_edges("insight_agent", route_next)
        workflow.add_conditional_edges("synthesis_agent", route_next)
        
        # Set entry point
        workflow.add_edge(START, "citizen_agent")
        
        return workflow
    
    def process_query(self, query: str, thread_id: str = "main") -> Dict[str, Any]:
        """Process a query through the unified system"""
        
        try:
            # Initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_type": "",
                "entities": {},
                "raw_data": {},
                "processed_data": {},
                "current_agent": "citizen_agent",
                "next_agent": "citizen_agent",
                "completed_agents": [],
                "analysis_results": {},
                "insights": [],
                "final_answer": "",
                "conversation_memory": [],
                "long_term_insights": {}
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
                "raw_data": result.get("raw_data", {}),
                "insights": result.get("insights", []),
                "memory_context": result.get("long_term_insights", {})
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_answer": f"I encountered an error: {str(e)}"
            }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated memory"""
        return {
            "query_history_count": len(self.connector.query_history),
            "recent_patterns": self.connector._analyze_query_patterns(),
            "system_status": "active"
        }


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class UnifiedApp:
    """Main application interface"""
    
    def __init__(self):
        self.system = None
    
    def initialize(self, db_path: str = "../db/vehicles.db") -> bool:
        """Initialize the system"""
        
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                print("❌ OPENAI_API_KEY not found!")
                return False
            
            print("🚀 Initializing Unified Vehicle Registration AI System...")
            self.system = UnifiedVehicleRegistrationSystem(db_path, openai_key)
            print("✅ System ready!")
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
🧠 **Unified Vehicle Registration AI**
LlamaIndex (Data) + LangGraph (Orchestration) + Long-term Memory

Ask any question about Telangana vehicle registration data!
Type 'quit' to exit, 'memory' to see memory status.
        """)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                if query.lower() == 'memory':
                    memory_summary = self.system.get_memory_summary()
                    print(f"\n💾 Memory Status: {json.dumps(memory_summary, indent=2)}")
                    continue
                
                if not query:
                    continue
                
                print("\n🔄 Processing through multi-agent workflow...")
                
                result = self.system.process_query(query)
                
                if result["success"]:
                    print(f"\n✅ **Final Answer:**")
                    print(f"{result['final_answer']}")
                    
                    print(f"\n🤖 **Agents Used:** {', '.join(result['agents_used'])}")
                    
                    if result.get("insights"):
                        print(f"💡 **Key Insights:** {len(result['insights'])} generated")
                else:
                    print(f"\n❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point"""
    app = UnifiedApp()
    
    if app.initialize():
        app.run_interactive()
    else:
        print("Failed to initialize. Please check your setup.")


if __name__ == "__main__":
    main()