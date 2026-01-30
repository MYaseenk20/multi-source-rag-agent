import streamlit as st

from backend.core import AgentService

st.set_page_config(
    page_title="LangChain Agent Chat",
    page_icon="🤖",
)

st.title("🤖 LangChain Agent Assistant")
st.markdown("Ask questions about LangChain and HR Policy!")

prompt = st.text_input("Prompt",placeholder="Enter your prompt here...")

if "agent_service" not in st.session_state:
    with st.spinner("🔄 Initializing agent..."):
        st.session_state["agent_service"] = AgentService()

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "user_answers_history" not in st.session_state:
    st.session_state["user_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Processing..."):
        generated_response = st.session_state.agent_service.process_chat(prompt,st.session_state.chat_history)
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["user_answers_history"].append(generated_response)
        st.session_state["chat_history"].append(("human",prompt))
        st.session_state["chat_history"].append(("ai",generated_response["result"]))

if st.session_state["user_answers_history"]:
    for generated_response,user_query in zip(st.session_state["user_answers_history"],st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)