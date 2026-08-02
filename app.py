import streamlit as st
from rag_engine import build_ai_engine

# --- Page Setup ---
st.set_page_config(page_title="World Data AI", page_icon="🌍")
st.title("🌍 Chat with the World Database")
st.markdown("I have memorized capitals, currencies, and religions for countries around the world. Ask me anything!")

# --- Load the AI Brain (Only happens once) ---
if "rag_chain" not in st.session_state:
    with st.spinner("Loading AI Brain and reading world data..."):
        # We import the exact function you wrote in rag_engine.py!
        st.session_state.rag_chain = build_ai_engine()
        st.success("World data perfectly loaded into the Vector Database!")

# --- Chat History Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am the Global AI Assistant. Which country would you like to know about?"}
    ]

# Print all previous messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Box ---
user_query = st.chat_input("E.g., What is the currency of Japan?")

if user_query:
    # 1. Print the user's question on the screen
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Ask the RAG Engine for the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching world records..."):
            # We pass the user's question into your LCEL pipeline!
            ai_answer = st.session_state.rag_chain.invoke(user_query)
            
            # Print the AI's answer on the screen
            st.markdown(ai_answer)
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})