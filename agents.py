from typing import Literal, TypedDict
import functools
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from tools import perform_clustering, generate_visualization, clean_data, perform_eda

# 1. Define LLM
# Using gpt-4o-mini as a robust default for orchestration
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Define Specialist Agents
cleaning_agent = create_react_agent(
    model=llm,
    tools=[clean_data, perform_eda],
    name="cleaning_agent",
    debug=True
)

clustering_agent = create_react_agent(
    model=llm,
    tools=[perform_clustering],
    name="clustering_agent",
    debug=True
)

visualization_agent = create_react_agent(
    model=llm,
    tools=[generate_visualization],
    name="visualization_agent",
    debug=True
)

# 3. Define the Supervisor Node
# This node decides who goes next based on the history
members = ["cleaning_agent", "clustering_agent", "visualization_agent"]
options = members + ["FINISH"]

def supervisor_node(state: AgentState):
    """
    Acts as the orchestrator. It receives the user's natural language request 
    and decides which specialist agent to call next.
    """
    messages = [
        SystemMessage(content=f"""You are a dedicated Autonomous Data Analytics Manager. 
        Your goal is to COMPLETELY execute the user's requested workflow without stopping for human input until the very end.
        
        Current members: {members}
        Active Data File: {state.get('df_path', 'None')}
        
        THE PLAN (Execute in Order):
        1. CHECK DATA: If nulls/outliers exist -> call cleaning_agent.
        2. ANALYZE: If user wants EDA -> call cleaning_agent (perform_eda).
        3. CLUSTER: If user requested clustering -> call clustering_agent. (CRITICAL: Do not skip this if requested!)
        4. VISUALIZE: If user requested plots/viz -> call visualization_agent.
        
        STRICT RULES:
        - CHECK THE CHAT HISTORY. Have you performed Clustering yet? If no, and the user asked for it, CALL clustering_agent.
        - Have you performed Visualization yet? If no, and the user asked for it, CALL visualization_agent.
        - Ignore any polite "Let me know/Ask me" text from the sub-agents. They are subordinates. You are the boss.
        - If the cleaning agent says "ready for clustering", YOU MUST CALL clustering_agent.
        - DO NOT CALL FINISH until you see a 'Cluster Visualization' or 'Scatter Plot' in the history.
        
        Reply ONLY with the name of the next agent or FINISH.""")
    ] + state["messages"]

    # Simple logic-based routing or LLM-based. Let's use LLM for "Supervisor Pattern"
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Validation with Robust Parsing
    # If the LLM is chatty (e.g., "I will call clustering_agent"), we need to extract the agent name.
    detected_agent = "FINISH"
    for option in options:
        if option in content:
            detected_agent = option
            break
            
    # Prefer explicit "FINISH" if it's the only option, but if an agent is mentioned, go there.
    # Logic: If content contains "clustering_agent", go there.
    if detected_agent == "FINISH" and "FINISH" not in content:
        # Fallback: If no valid option is found in the text, default to FINISH (or handle error)
        # But let's look for partial matches or just force strictness. 
        # Actually, let's trust the "detected_agent" loop.
        pass
    
    return {"next_node": detected_agent}

# 4. Agent Execution Helpers
def run_specialist(state: AgentState, name: str, agent):
    """Bridge function to run a specialist and update the global state."""
    # Inject ABSOLUTE data file path and instructions to return control to supervisor
    current_path = state.get("df_path", "No file uploaded")
    
    # We add a very specific instruction to break the ReACT loop once tools are called
    # but without implying that the entire project is over.
    context_msg = SystemMessage(content=f"""
    IMPORTANT: The currently active data file path is: {current_path}. 
    This is an ABSOLUTE PATH. You MUST use this EXACT string for all tool 'file_path' arguments.
    
    INSTRUCTIONS:
    1. If you use 'clean_data', it will save a new file (identifiable by '_cleaned' in the path).
    2. If you see high correlations in 'perform_eda', use 'clean_data' ONCE more with 'drop_columns' (using the SUGGESTED DROPS from the EDA report) to fix them.
    3. Once you have performed your specific task (e.g. cleaning, EDA, or clustering), summarize what was done and then FINISH YOUR TURN.
    4. REPORT: "Task Complete. Data is ready for [Next Step]."
    5. FORBIDDEN: Do NOT ask the user "How would you like to proceed?". Do NOT saying "Let me know". Just report facts and exit.
    6. DO NOT loop indefinitely. Perform the action, summarize, and exit.
    """)
    
    # Create a local state with the injected message
    local_state = state.copy()
    local_state["messages"] = list(local_state["messages"]) + [context_msg]
    
    # Record current message count to extract ONLY new ones later
    history_len = len(local_state["messages"])
    
    # Use a high recursion limit for the session
    config = {"recursion_limit": 50}
    result = agent.invoke(local_state, config=config)
    
    # Extract ONLY new messages added by the specialist
    new_messages = result["messages"][history_len:]
    
    # Extract file path from tool output if possible (detecting absolute paths)
    df_path = state.get("df_path", "")
    for m in reversed(new_messages):
        if hasattr(m, "content"):
            last_msg = m.content
            if "saved to:" in last_msg.lower():
                import re
                # Match both unix and windows paths (detects .csv at end)
                match = re.search(r"saved to: (.*?\.csv)", last_msg)
                if match:
                    df_path = match.group(1).strip()
                    break

    return {
        "messages": new_messages,
        "df_path": df_path
    }

# Bind agents to the runner
cleaning_node = functools.partial(run_specialist, name="cleaning_agent", agent=cleaning_agent)
clustering_node = functools.partial(run_specialist, name="clustering_agent", agent=clustering_agent)
visualization_node = functools.partial(run_specialist, name="visualization_agent", agent=visualization_agent)

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("cleaning_agent", cleaning_node)
workflow.add_node("clustering_agent", clustering_node)
workflow.add_node("visualization_agent", visualization_node)

# Entry point
workflow.set_entry_point("supervisor")

# Conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {
        "cleaning_agent": "cleaning_agent",
        "clustering_agent": "clustering_agent",
        "visualization_agent": "visualization_agent",
        "FINISH": END
    }
)

# Return to supervisor after each agent
workflow.add_edge("cleaning_agent", "supervisor")
workflow.add_edge("clustering_agent", "supervisor")
workflow.add_edge("visualization_agent", "supervisor")

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
