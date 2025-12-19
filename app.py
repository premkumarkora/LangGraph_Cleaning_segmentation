import streamlit as st
import pandas as pd
import os
import uuid
import plotly.graph_objects as go
import json
from agents import graph
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("Multi-Agent Data Analytics Supervisor")
st.markdown("""
Upload a CSV and ask your AI Supervisor to perform clustering or visualizations.
The supervisor orchestrated specialists (`Clustering Agent` & `Visualization Agent`) to get the job done.
""")

# 2. Sidebar - Configuration & Upload
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    
    if uploaded_file:
        # Save locally to be accessible by tools
        temp_dir = os.path.abspath("temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")
        st.session_state.df_path = file_path
    else:
        st.session_state.df_path = None

# 3. Main Area - Data Preview
if st.session_state.df_path:
    df = pd.read_csv(st.session_state.df_path)
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Rows: {df.shape[0]} | Columns: {list(df.columns)}")

st.divider()

# 4. Chat Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "auto_triggered" not in st.session_state:
    st.session_state.auto_triggered = False

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process_request(prompt: str):
    """Refactored agent execution logic for both manual and auto triggers."""
    # User message
    if not any(m["content"] == prompt for m in st.session_state.messages):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Agent Execution with Streaming Trace
    with st.chat_message("assistant"):
        trace_placeholder = st.expander("Agent Thinking Process (Trace)", expanded=True)
        response_placeholder = st.empty()
        
        # Initial state
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 100
        }
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "df_path": st.session_state.df_path
        }

        final_output = ""
        
        # Stream the graph execution
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, output in event.items():
                # Trace output
                with trace_placeholder:
                    if node_name == "supervisor":
                        next_node = output.get("next_node", "Unknown")
                        st.write(f"**Supervisor:** Decided to call `{next_node}`")
                    
                    elif node_name == "cleaning_agent":
                        st.write("🧹 **Cleaning Agent:** Analyzing data quality...")
                        if "df_path" in output:
                            st.session_state.df_path = output["df_path"]
                        # Display Summary in Main Chat
                        if "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                st.markdown(f"### 🧹 Cleaning & EDA Report\n{last_msg.content}")
                                st.divider()

                    elif node_name == "clustering_agent":
                        st.write("**Clustering Agent:** Preparing for K-Means analysis...")
                        if "df_path" in output:
                            st.session_state.df_path = output["df_path"]
                        # Display Summary in Main Chat
                        if "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                st.markdown(f"### Clustering Report\n{last_msg.content}")
                                st.divider()

                    elif node_name == "visualization_agent":
                        st.write("🎨 **Visualization Agent:** Initiating Plotly rendering...")
                        # Display Summary in Main Chat
                        if "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                st.markdown(f"### 🎨 Visualization Report\n{last_msg.content}")
                                st.divider()
                
                # Store messages and show tool calls in trace
                if "messages" in output:
                    for m in output["messages"]:
                        if isinstance(m, AIMessage):
                            final_output = m.content
                            # Check for tool calls and show them in trace
                            if hasattr(m, 'tool_calls') and m.tool_calls:
                                for tc in m.tool_calls:
                                    with trace_placeholder:
                                        st.code(f"🔨 Calling Tool: {tc['name']}\nArguments: {json.dumps(tc['args'], indent=2)}", language="json")

        # Final response
        if final_output:
            response_placeholder.markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})
        else:
            response_placeholder.markdown("Task complete.")

        # --- DYNAMIC RENDERING BASED ON AGENT ACTIONS ---
        
        # 1. EDA Visualizations
        if "signals generated for ui" in final_output.lower():
            st.divider()
            st.subheader("Exploratory Data Analysis")
            df_eda = pd.read_csv(st.session_state.df_path)
            num_df = df_eda.select_dtypes(include=[np.number])
            
            if not num_df.empty:
                tab1, tab2 = st.tabs(["Correlation Heatmap", "Feature Distributions"])
                with tab1:
                    corr = num_df.corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
                    st.plotly_chart(fig_corr, use_container_width=True)
                with tab2:
                    sel_col = st.selectbox("Select column to view distribution", num_df.columns, key="eda_sel")
                    fig_dist = px.histogram(num_df, x=sel_col, marginal="box", title=f"Distribution of {sel_col}", template="plotly_dark")
                    st.plotly_chart(fig_dist, use_container_width=True)

        # 2. Cluster Visualizations
        if st.session_state.df_path and "_clustered.csv" in st.session_state.df_path:
            st.divider()
            st.subheader("Cluster Analysis")
            df_viz = pd.read_csv(st.session_state.df_path)
            import plotly.express as px
            fig = px.scatter(
                df_viz, x="PCA1", y="PCA2", color="Cluster", 
                title="2D PCA Cluster Map",
                template="plotly_dark",
                hover_data=df_viz.columns.tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success("Analysis and Visualization Complete!")
            
            # Allow user to download the final result
            with open(st.session_state.df_path, "rb") as f:
                st.download_button(
                    label="Download Processed Data",
                    data=f,
                    file_name=os.path.basename(st.session_state.df_path),
                    mime="text/csv"
                )

# 5. Autonomous / Manual Trigger
if st.session_state.df_path and not st.session_state.auto_triggered:
    st.info("Data Uploaded. Starting Autonomous Analysis Pipeline...")
    st.session_state.auto_triggered = True
    # Define a comprehensive default prompt
    default_prompt = "Perform full data cleaning (nulls, outliers), exploratory analysis (EDA), and then cluster all numeric features into 3 groups and visualize."
    process_request(default_prompt)

if prompt := st.chat_input("Ask a follow-up question..."):
    process_request(prompt)

# 6. Sidebar - Debug/Session Info
with st.sidebar:
    st.divider()
    st.header("Debug Info")
    st.write(f"**Current CSV:** {os.path.basename(st.session_state.df_path) if st.session_state.df_path else 'None'}")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
