"""
Streamlit App for Vehicle Registration AI System
===============================================

A user-friendly interface to interact with the enhanced Vehicle Registration AI system
that combines LlamaIndex, LangGraph, and long-term memory for intelligent data analysis.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

# Import the integrated system
from agents.langgraph_integration import IntegratedVehicleRegistrationSystem

# Page configuration
st.set_page_config(
    page_title="🚗 Vehicle Registration AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .ai-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .system-message {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system" not in st.session_state:
    st.session_state.system = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def initialize_system():
    """Initialize the integrated system"""
    try:
        with st.spinner("🚀 Initializing Vehicle Registration AI System..."):
            system = IntegratedVehicleRegistrationSystem()
            system.initialize()
            st.session_state.system = system
            st.success("✅ System initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        return False

def display_chat_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>🤖 AI Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div class="chat-message system-message">
            <strong>⚙️ System:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    # Display metadata if available
    if metadata:
        with st.expander("📊 Query Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{metadata.get('confidence', 0):.1f}%")
            with col2:
                st.metric("Agents Used", len(metadata.get('agents_used', [])))
            with col3:
                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
            
            if metadata.get('agents_used'):
                st.write("**Agents Used:**", ", ".join(metadata['agents_used']))
            
            if metadata.get('sql_query'):
                st.code(metadata['sql_query'], language='sql')

def process_query(query: str) -> Dict[str, Any]:
    """Process a user query through the integrated system"""
    if not st.session_state.system:
        return {"error": "System not initialized"}
    
    try:
        start_time = time.time()
        
        # Process the query
        result = st.session_state.system.process_query(query)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "answer": result.get("final_answer", "No answer provided"),
            "confidence": result.get("confidence", 0),
            "agents_used": result.get("agents_used", []),
            "processing_time": processing_time,
            "sql_query": result.get("sql_query"),
            "metadata": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Vehicle Registration AI System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Intelligent analysis of Telangana vehicle registration data using LlamaIndex, LangGraph, and AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
        
        # System status
        if st.session_state.system:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("📊 Show Database Stats"):
            if st.session_state.system:
                with st.spinner("Fetching database statistics..."):
                    result = process_query("How many vehicles are registered in total?")
                    if result.get("success"):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "metadata": result
                        })
        
        if st.button("🔍 Sample Queries"):
            sample_queries = [
                "How many vehicles are registered in 2024?",
                "What are the top 5 vehicle manufacturers?",
                "How many vehicles use petrol vs diesel?",
                "What are the pollution levels from vehicles?",
                "What changes can we make to improve transportation?"
            ]
            st.session_state.sample_queries = sample_queries
        
        st.divider()
        
        # Sample queries
        if hasattr(st.session_state, 'sample_queries'):
            st.header("💡 Sample Queries")
            for query in st.session_state.sample_queries:
                if st.button(f"📝 {query[:50]}...", key=f"sample_{query}"):
                    st.session_state.user_input = query
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("metadata")
            )
        
        # Chat input
        user_input = st.text_input(
            "Ask a question about vehicle registration data:",
            placeholder="e.g., How many vehicles are registered in Hyderabad?",
            key="user_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            send_button = st.button("Send", type="primary")
        
        # Process query
        if send_button and user_input and st.session_state.system:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Process query
            with st.spinner("🤔 Processing your query..."):
                result = process_query(user_input)
            
            if result.get("success"):
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result
                })
                
                # Update conversation history
                st.session_state.conversation_history.extend([
                    user_input,
                    result["answer"]
                ])
            else:
                # Add error message
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"❌ Error: {result.get('error', 'Unknown error')}"
                })
            
            st.rerun()
    
    with col2:
        st.header("📈 System Information")
        
        # Database info
        st.subheader("🗄️ Database")
        st.info("""
        **Telangana Vehicle Registration Database**
        - Total Records: 1,880,183
        - Data Fields: Registration details, vehicle specs, fuel types
        - Coverage: Multiple years of registration data
        """)
        
        # System capabilities
        st.subheader("🧠 AI Capabilities")
        st.info("""
        **Smart Query Processing:**
        - Data queries → SQL + LlamaIndex
        - Analytical questions → LLM reasoning
        - Follow-up questions → Context-aware analysis
        - Long-term memory → Conversation history
        """)
        
        # Performance metrics
        if st.session_state.messages:
            st.subheader("📊 Session Stats")
            total_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
            avg_confidence = sum([
                m.get("metadata", {}).get("confidence", 0) 
                for m in st.session_state.messages 
                if m.get("metadata", {}).get("confidence")
            ]) / max(total_queries, 1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Tips
        st.subheader("💡 Tips")
        st.info("""
        **Best Practices:**
        - Ask specific data questions for SQL queries
        - Use follow-up questions for analysis
        - Try analytical questions for insights
        - The system remembers conversation context
        """)

if __name__ == "__main__":
    main()
