import streamlit as st
from backend.core import AgentService

if "agent_service" not in st.session_state:
    with st.spinner("🔄 Initializing agent..."):
        st.session_state["agent_service"] = AgentService()

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "user_answers_history" not in st.session_state:
    st.session_state["user_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.set_page_config(
    page_title="LangChain Agent Chat",
    page_icon="🤖",
)

st.title("🤖 LangChain Agent Assistant")
st.markdown("Ask questions about LangChain and HR Policy!")

# Display chat history first
if st.session_state["user_answers_history"]:
    for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["user_answers_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

# Chat input at the bottom (automatically clears after submission)
prompt = st.chat_input("Enter your prompt here...")

if prompt:
    with st.spinner("Processing..."):
        generated_response = st.session_state["agent_service"].process_chat(prompt, st.session_state.chat_history)
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["user_answers_history"].append(generated_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response))
        st.rerun()