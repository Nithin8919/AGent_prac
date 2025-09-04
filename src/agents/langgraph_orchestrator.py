"""
LangGraph Multi-Agent Orchestrator for Vehicle Registration Analysis
===================================================================

This is the core orchestrator that creates truly intelligent multi-agent workflows
for complex policy analysis, infrastructure planning, and strategic decision-making.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from typing_extensions import NotRequired
import operator
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from loguru import logger


class AnalysisType(Enum):
    """Types of complex analysis the system can perform"""
    POLICY_IMPACT = "policy_impact"
    INFRASTRUCTURE_PLANNING = "infrastructure_planning"  
    ECONOMIC_ANALYSIS = "economic_analysis"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    TREND_PREDICTION = "trend_prediction"
    SCENARIO_MODELING = "scenario_modeling"
    COMPARATIVE_STUDY = "comparative_study"


class AgentState(TypedDict):
    """Shared state across all agents in the workflow"""
    # Core conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Analysis context
    analysis_type: str
    original_query: str
    complexity_score: float
    
    # Workflow management
    current_agent: str
    next_agent: str
    workflow_plan: List[str]
    completed_agents: List[str]
    
    # Data and results
    raw_data: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]]
    
    # Quality control
    validation_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    human_review_required: bool
    
    # Context and memory
    conversation_history: List[Dict[str, Any]]
    long_term_insights: List[str]
    user_preferences: Dict[str, Any]
    
    # Final output
    final_analysis: NotRequired[Dict[str, Any]]
    recommendations: NotRequired[List[str]]
    executive_summary: NotRequired[str]


@dataclass
class WorkflowPlan:
    """Represents a multi-step analysis plan"""
    analysis_type: str
    required_agents: List[str]
    agent_dependencies: Dict[str, List[str]]
    estimated_complexity: float
    expected_duration: str
    quality_checkpoints: List[str]


class SupervisorAgent:
    """
    Orchestrator Agent - Plans and coordinates complex multi-agent workflows
    This is the brain that decides what needs to be analyzed and how
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "SupervisorAgent"
        
        # Available specialized agents
        self.available_agents = {
            "planning_agent": "Decomposes complex questions into analysis plans",
            "data_agent": "Retrieves and processes vehicle registration data", 
            "policy_agent": "Analyzes policy implications and regulatory impacts",
            "economic_agent": "Performs economic impact and cost-benefit analysis",
            "environmental_agent": "Assesses environmental and sustainability impacts",
            "infrastructure_agent": "Evaluates infrastructure needs and planning",
            "prediction_agent": "Creates forecasts and scenario modeling",
            "synthesis_agent": "Combines analyses into comprehensive insights",
            "validation_agent": "Quality assurance and fact-checking"
        }
        
        # Workflow templates for common analysis types
        self.workflow_templates = {
            AnalysisType.POLICY_IMPACT: [
                "planning_agent", "data_agent", "policy_agent", 
                "economic_agent", "environmental_agent", "synthesis_agent", "validation_agent"
            ],
            AnalysisType.INFRASTRUCTURE_PLANNING: [
                "planning_agent", "data_agent", "infrastructure_agent",
                "prediction_agent", "economic_agent", "synthesis_agent", "validation_agent"
            ],
            AnalysisType.SCENARIO_MODELING: [
                "planning_agent", "data_agent", "prediction_agent",
                "policy_agent", "economic_agent", "environmental_agent", 
                "synthesis_agent", "validation_agent"
            ]
        }
        
        logger.info(f"✅ {self.name} initialized with {len(self.available_agents)} specialist agents")

    def __call__(self, state: AgentState) -> AgentState:
        """Main supervisor logic - decides workflow and next steps"""
        
        try:
            # Get the last message
            last_message = state["messages"][-1] if state["messages"] else None
            
            if not last_message:
                return self._initialize_workflow(state)
            
            # If this is a new query, create a workflow plan
            if state.get("current_agent") == "supervisor" or not state.get("workflow_plan"):
                return self._create_workflow_plan(state, last_message.content)
            
            # If workflow is in progress, check status and route
            return self._manage_ongoing_workflow(state)
            
        except Exception as e:
            logger.error(f"Supervisor agent failed: {e}")
            state["messages"].append(AIMessage(
                content=f"I encountered an error planning the analysis: {str(e)}. Let me try a simpler approach."
            ))
            return state

    def _initialize_workflow(self, state: AgentState) -> AgentState:
        """Initialize the workflow system"""
        state["current_agent"] = "supervisor"
        state["completed_agents"] = []
        state["agent_outputs"] = {}
        state["confidence_scores"] = {}
        state["conversation_history"] = []
        
        welcome_msg = """🧠 **Intelligent Policy Analysis System Ready**

I can help you with sophisticated analysis of vehicle registration data including:

🔍 **Policy Impact Analysis**: "What would happen if we banned diesel vehicles?"
🏗️ **Infrastructure Planning**: "Where should we build EV charging stations?" 
📈 **Economic Analysis**: "What's the economic impact of EV adoption?"
🌱 **Environmental Studies**: "How would EV growth affect air quality?"
📊 **Trend Predictions**: "What will vehicle registrations look like in 2030?"
🔄 **Scenario Modeling**: "Compare different fuel policy scenarios"

What complex analysis would you like me to perform?"""

        state["messages"].append(AIMessage(content=welcome_msg))
        return state

    def _create_workflow_plan(self, state: AgentState, query: str) -> AgentState:
        """Create a comprehensive workflow plan for the query"""
        
        # Classify the analysis type and complexity
        analysis_classification = self._classify_query(query)
        
        state["analysis_type"] = analysis_classification["type"]
        state["original_query"] = query
        state["complexity_score"] = analysis_classification["complexity"]
        
        # Create workflow plan
        workflow_plan = self._generate_workflow_plan(analysis_classification)
        
        state["workflow_plan"] = workflow_plan.required_agents
        state["next_agent"] = workflow_plan.required_agents[0]
        
        # Create planning message
        planning_msg = f"""📋 **Analysis Plan Created**

**Query**: {query}
**Analysis Type**: {analysis_classification['type'].replace('_', ' ').title()}
**Complexity Score**: {analysis_classification['complexity']:.1f}/5.0
**Estimated Duration**: {workflow_plan.expected_duration}

**Execution Plan**:
{self._format_workflow_plan(workflow_plan)}

🚀 **Starting analysis with {workflow_plan.required_agents[0]}...**
"""
        
        state["messages"].append(AIMessage(content=planning_msg))
        
        # Store workflow metadata
        state["agent_outputs"]["supervisor"] = {
            "workflow_plan": asdict(workflow_plan),
            "analysis_classification": analysis_classification,
            "timestamp": datetime.now().isoformat()
        }
        
        return state

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to classify query type and complexity"""
        
        classification_prompt = f"""
        Analyze this query about vehicle registration data and classify it:
        
        Query: "{query}"
        
        Determine:
        1. Analysis Type: policy_impact, infrastructure_planning, economic_analysis, environmental_impact, trend_prediction, scenario_modeling, or comparative_study
        2. Complexity (1-5): How many analysis steps and data points are needed?
        3. Key Components: What aspects need to be analyzed?
        4. Stakeholders: Who would be interested in this analysis?
        
        Respond in JSON format:
        {{
            "type": "analysis_type",
            "complexity": 3.5,
            "components": ["component1", "component2"],
            "stakeholders": ["stakeholder1", "stakeholder2"],
            "reasoning": "why this classification"
        }}
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=classification_prompt)])
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return classification
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
        
        # Fallback classification
        return {
            "type": "policy_impact",
            "complexity": 3.0,
            "components": ["data_analysis", "impact_assessment"],
            "stakeholders": ["policy_makers"],
            "reasoning": "Fallback classification due to parsing error"
        }

    def _generate_workflow_plan(self, classification: Dict[str, Any]) -> WorkflowPlan:
        """Generate detailed workflow plan based on classification"""
        
        analysis_type_enum = AnalysisType(classification["type"])
        
        # Get base workflow from template
        base_agents = self.workflow_templates.get(
            analysis_type_enum, 
            ["planning_agent", "data_agent", "synthesis_agent", "validation_agent"]
        )
        
        # Customize based on complexity and components
        complexity = classification["complexity"]
        components = classification.get("components", [])
        
        # Add specialized agents based on components
        if "economic" in str(components).lower() and "economic_agent" not in base_agents:
            base_agents.insert(-2, "economic_agent")
        
        if "environment" in str(components).lower() and "environmental_agent" not in base_agents:
            base_agents.insert(-2, "environmental_agent")
        
        if "predict" in str(components).lower() and "prediction_agent" not in base_agents:
            base_agents.insert(-2, "prediction_agent")
        
        # Create dependencies
        dependencies = {}
        for i, agent in enumerate(base_agents):
            if i == 0:
                dependencies[agent] = []  # First agent has no dependencies
            else:
                dependencies[agent] = [base_agents[i-1]]  # Each depends on previous
        
        # Special dependencies
        if "synthesis_agent" in base_agents:
            # Synthesis depends on all analysis agents
            analysis_agents = [a for a in base_agents if a not in ["planning_agent", "synthesis_agent", "validation_agent"]]
            dependencies["synthesis_agent"] = analysis_agents
        
        if "validation_agent" in base_agents:
            dependencies["validation_agent"] = ["synthesis_agent"]
        
        # Estimate duration based on complexity
        duration_map = {
            1.0: "1-2 minutes",
            2.0: "2-3 minutes", 
            3.0: "3-5 minutes",
            4.0: "5-8 minutes",
            5.0: "8-15 minutes"
        }
        
        estimated_duration = duration_map.get(
            int(complexity), 
            f"{int(complexity * 2)}-{int(complexity * 3)} minutes"
        )
        
        return WorkflowPlan(
            analysis_type=classification["type"],
            required_agents=base_agents,
            agent_dependencies=dependencies,
            estimated_complexity=complexity,
            expected_duration=estimated_duration,
            quality_checkpoints=["validation_agent"]
        )

    def _format_workflow_plan(self, plan: WorkflowPlan) -> str:
        """Format workflow plan for display"""
        
        formatted_steps = []
        for i, agent in enumerate(plan.required_agents, 1):
            agent_desc = self.available_agents.get(agent, agent)
            formatted_steps.append(f"   {i}. **{agent.replace('_', ' ').title()}**: {agent_desc}")
        
        return "\n".join(formatted_steps)

    def _manage_ongoing_workflow(self, state: AgentState) -> AgentState:
        """Manage ongoing workflow and decide next steps"""
        
        current_agent = state.get("current_agent", "")
        completed_agents = state.get("completed_agents", [])
        workflow_plan = state.get("workflow_plan", [])
        
        # Check if current agent is completed
        if current_agent and current_agent not in completed_agents:
            completed_agents.append(current_agent)
            state["completed_agents"] = completed_agents
        
        # Find next agent
        remaining_agents = [a for a in workflow_plan if a not in completed_agents]
        
        if remaining_agents:
            next_agent = remaining_agents[0]
            state["next_agent"] = next_agent
            state["current_agent"] = next_agent
            
            progress_msg = f"✅ **{current_agent.replace('_', ' ').title()} Complete**\n\n🔄 **Next**: {next_agent.replace('_', ' ').title()} ({len(completed_agents)}/{len(workflow_plan)} agents completed)"
            
            state["messages"].append(AIMessage(content=progress_msg))
        else:
            # Workflow complete
            state["next_agent"] = "complete"
            state["current_agent"] = "supervisor"
            
            completion_msg = f"""🎉 **Analysis Complete!**

All {len(completed_agents)} specialist agents have completed their analysis.
Preparing comprehensive final report..."""
            
            state["messages"].append(AIMessage(content=completion_msg))
        
        return state


class PlanningAgent:
    """
    Strategic Planning Agent - Decomposes complex questions into detailed analysis plans
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "PlanningAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Create detailed analysis plan and research questions"""
        
        try:
            original_query = state.get("original_query", "")
            analysis_type = state.get("analysis_type", "")
            
            planning_prompt = f"""
            You are a strategic planning expert analyzing vehicle registration data for policy decisions.
            
            Original Query: "{original_query}"
            Analysis Type: {analysis_type}
            
            Create a comprehensive analysis plan that includes:
            1. Key research questions to answer
            2. Required data points and metrics
            3. Stakeholder considerations
            4. Success criteria for the analysis
            5. Potential limitations and assumptions
            6. Risk factors to consider
            
            Structure your response as a detailed planning document.
            """
            
            response = self.llm.invoke([
                SystemMessage(content=planning_prompt),
                HumanMessage(content=original_query)
            ])
            
            # Store planning results
            planning_results = {
                "detailed_plan": response.content,
                "research_questions": self._extract_research_questions(response.content),
                "required_data": self._extract_data_requirements(response.content),
                "timestamp": datetime.now().isoformat()
            }
            
            state["agent_outputs"]["planning_agent"] = planning_results
            state["confidence_scores"]["planning_agent"] = 0.9
            
            planning_msg = f"""📋 **Strategic Analysis Plan**

{response.content}

**Research Questions Identified**: {len(planning_results['research_questions'])}
**Data Requirements**: {len(planning_results['required_data'])} key datasets

✅ **Planning Complete** - Ready for data analysis phase"""

            state["messages"].append(AIMessage(content=planning_msg))
            
            return state
            
        except Exception as e:
            logger.error(f"Planning agent failed: {e}")
            state["messages"].append(AIMessage(
                content=f"Planning analysis encountered an error: {str(e)}"
            ))
            return state
    
    def _extract_research_questions(self, plan_text: str) -> List[str]:
        """Extract research questions from planning text"""
        # Simple extraction - in production would use more sophisticated parsing
        import re
        questions = re.findall(r'\d+\.\s*([^?\n]+\?)', plan_text)
        return questions[:10]  # Limit to top 10
    
    def _extract_data_requirements(self, plan_text: str) -> List[str]:
        """Extract data requirements from planning text"""
        # Simple extraction
        import re
        data_mentions = re.findall(r'(?:data|dataset|metric|measurement|statistic)s?\s+(?:on|for|about)\s+([^.\n]+)', plan_text.lower())
        return list(set(data_mentions))[:10]


class VehicleRegistrationOrchestrator:
    """
    Main orchestrator class that creates and manages the LangGraph workflow
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize the complete multi-agent system"""
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # Initialize memory for persistent conversations
        self.memory = MemorySaver()
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.llm)
        self.planning_agent = PlanningAgent(self.llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow_graph()
        
        # Compile the graph with memory
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("🚀 Vehicle Registration Multi-Agent Orchestrator initialized")

    def _build_workflow_graph(self) -> StateGraph:
        """Build the complete LangGraph workflow"""
        
        # Create the graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("supervisor", self.supervisor)
        workflow.add_node("planning_agent", self.planning_agent)
        
        # Add placeholder nodes for other agents (to be implemented)
        workflow.add_node("data_agent", self._placeholder_agent("DataAgent"))
        workflow.add_node("policy_agent", self._placeholder_agent("PolicyAgent"))
        workflow.add_node("economic_agent", self._placeholder_agent("EconomicAgent"))
        workflow.add_node("environmental_agent", self._placeholder_agent("EnvironmentalAgent"))
        workflow.add_node("infrastructure_agent", self._placeholder_agent("InfrastructureAgent"))
        workflow.add_node("prediction_agent", self._placeholder_agent("PredictionAgent"))
        workflow.add_node("synthesis_agent", self._placeholder_agent("SynthesisAgent"))
        workflow.add_node("validation_agent", self._placeholder_agent("ValidationAgent"))
        
        # Define the routing logic
        def route_next_agent(state: AgentState) -> str:
            """Dynamic routing based on workflow state"""
            next_agent = state.get("next_agent", "supervisor")
            
            if next_agent == "complete":
                return END
            elif next_agent in ["supervisor", "planning_agent", "data_agent", "policy_agent", 
                               "economic_agent", "environmental_agent", "infrastructure_agent",
                               "prediction_agent", "synthesis_agent", "validation_agent"]:
                return next_agent
            else:
                return "supervisor"  # Default fallback
        
        # Set up edges
        workflow.add_conditional_edges("supervisor", route_next_agent)
        workflow.add_conditional_edges("planning_agent", route_next_agent)
        workflow.add_conditional_edges("data_agent", route_next_agent)
        workflow.add_conditional_edges("policy_agent", route_next_agent)
        workflow.add_conditional_edges("economic_agent", route_next_agent)
        workflow.add_conditional_edges("environmental_agent", route_next_agent)
        workflow.add_conditional_edges("infrastructure_agent", route_next_agent)
        workflow.add_conditional_edges("prediction_agent", route_next_agent)
        workflow.add_conditional_edges("synthesis_agent", route_next_agent)
        workflow.add_conditional_edges("validation_agent", route_next_agent)
        
        # Set entry point
        workflow.add_edge(START, "supervisor")
        
        return workflow

    def _placeholder_agent(self, agent_name: str):
        """Create placeholder agent functions for agents not yet implemented"""
        
        def placeholder(state: AgentState) -> AgentState:
            state["messages"].append(AIMessage(
                content=f"🔧 **{agent_name}** is processing your request... (Implementation in progress)\n\n✅ **Mock Analysis Complete** - Moving to next agent."
            ))
            
            # Store mock results
            state["agent_outputs"][agent_name.lower() + "_agent"] = {
                "status": "placeholder_completed",
                "timestamp": datetime.now().isoformat()
            }
            state["confidence_scores"][agent_name.lower() + "_agent"] = 0.8
            
            return state
            
        return placeholder

    def process_query(self, query: str, thread_id: str = "main") -> Dict[str, Any]:
        """Process a query through the complete multi-agent workflow"""
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "analysis_type": "",
                "original_query": query,
                "complexity_score": 0.0,
                "current_agent": "supervisor",
                "next_agent": "supervisor", 
                "workflow_plan": [],
                "completed_agents": [],
                "raw_data": {},
                "agent_outputs": {},
                "intermediate_results": [],
                "validation_results": {},
                "confidence_scores": {},
                "human_review_required": False,
                "conversation_history": [],
                "long_term_insights": [],
                "user_preferences": {}
            }
            
            # Configure thread for memory
            config = {"configurable": {"thread_id": thread_id}}
            
            # Run the workflow
            result = self.app.invoke(initial_state, config=config)
            
            # Format response
            return {
                "success": True,
                "messages": [msg.content for msg in result["messages"]],
                "workflow_completed": result.get("next_agent") == "complete",
                "agent_outputs": result.get("agent_outputs", {}),
                "confidence_scores": result.get("confidence_scores", {}),
                "analysis_type": result.get("analysis_type", ""),
                "final_state": result
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "messages": [f"I encountered an error processing your query: {str(e)}"]
            }

    def get_conversation_state(self, thread_id: str = "main") -> Dict[str, Any]:
        """Get current conversation state for a thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            return {
                "thread_id": thread_id,
                "current_agent": state.values.get("current_agent", ""),
                "completed_agents": state.values.get("completed_agents", []),
                "workflow_plan": state.values.get("workflow_plan", []),
                "message_count": len(state.values.get("messages", [])),
                "analysis_type": state.values.get("analysis_type", "")
            }
        except:
            return {"error": "Could not retrieve conversation state"}

    def stream_workflow(self, query: str, thread_id: str = "main"):
        """Stream the workflow execution for real-time updates"""
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "analysis_type": "",
            "original_query": query,
            "complexity_score": 0.0,
            "current_agent": "supervisor",
            "next_agent": "supervisor",
            "workflow_plan": [],
            "completed_agents": [],
            "raw_data": {},
            "agent_outputs": {},
            "intermediate_results": [],
            "validation_results": {},
            "confidence_scores": {},
            "human_review_required": False,
            "conversation_history": [],
            "long_term_insights": [],
            "user_preferences": {}
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream execution
        for chunk in self.app.stream(initial_state, config=config):
            yield chunk


# Factory function to create the orchestrator
def create_orchestrator(openai_api_key: str) -> VehicleRegistrationOrchestrator:
    """Create and return a configured orchestrator instance"""
    return VehicleRegistrationOrchestrator(openai_api_key)