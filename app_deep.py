import streamlit as st
import pandas as pd
import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Import the deep agent generator
from main_deepagents import get_deep_agent
from scripts.config import AGENT_USER_ID

# Set up the page
st.set_page_config(page_title="Deep Finance Researcher", page_icon="🧠", layout="wide")

st.title("🧠 Deep Finance Researcher AI")
st.markdown("""
Welcome to the **Deep Agent Mode**. 
*Optimized for Token Efficiency: Pruning historical memory to stay within daily rate limits.*
""")

# --- HELPER FUNCTION FOR CHARTS ---
def display_content(content):
    if isinstance(content, str) and "CHART_DATA|" in content:
        try:
            parts = content.split("|")
            chart_type = parts[1]
            title = parts[2]
            data_json = parts[3]
            
            df = pd.DataFrame(json.loads(data_json))
            if 'Label' in df.columns:
                df = df.set_index('Label')
            
            st.subheader(title)
            if chart_type.lower() == "line":
                st.line_chart(df)
            else:
                st.bar_chart(df)
        except Exception as e:
            st.error(f"Failed to render chart: {e}")
            st.code(content)
    else:
        st.markdown(content)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        display_content(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a financial question..."):
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Deep reasoning in progress... Planning, searching, and charting."):
            try:
                # 🔴 OPTIMIZATION: Generate a unique ID for this specific turn
                # This stops the SQLite database from loading the massive historical snowball
                current_interaction_id = f"turn_{uuid.uuid4().hex[:8]}"
                
                # 1. Boot up the deep agent with the temporary ID
                agent = get_deep_agent(AGENT_USER_ID, current_interaction_id)
                
                # 🔴 OPTIMIZATION: Grab ONLY the last 4 messages from Streamlit memory
                pruned_history = st.session_state.messages[-4:]
                
                # Convert Streamlit history to LangChain message formats
                input_messages = []
                for m in pruned_history:
                    # Skip chart data in history to save massive amounts of tokens
                    if "CHART_DATA|" in m["content"]:
                        continue 
                        
                    if m["role"] == "user":
                        input_messages.append(HumanMessage(content=m["content"]))
                    else:
                        input_messages.append(AIMessage(content=m["content"]))
                
                # 2. Run the agent with strictly limited memory
                response = agent.invoke(
                    {"messages": input_messages, "user_id": AGENT_USER_ID, "thread_id": current_interaction_id},
                    config={"configurable": {"thread_id": current_interaction_id}}
                )
                
                # 3. Extract final message
                final_answer = response["messages"][-1].content
                
                # 4. Show it on screen
                display_content(final_answer)
                
                # 5. Save to session state
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                st.error(f"An error occurred while running the deep agent: {str(e)[:500]}")