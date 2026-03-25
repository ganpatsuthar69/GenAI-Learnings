import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chatbot Experience")
st.markdown("A LangChain-powered chatbot using `deepseek-ai/DeepSeek-R1` on HuggingFace.")

with st.sidebar:
    st.header("Settings")
    mode_selection = st.radio(
        "Choose your AI mode:",
        options=["Helpful", "Angry", "Funny", "Sad"],
        index=0
    )
    
    if mode_selection == "Angry":
        system_mode = "You are an angry AI agent. You respond aggressively and impatiently."
    elif mode_selection == "Funny":
        system_mode = "You are a very funny AI agent. You respond with humor and jokes."
    elif mode_selection == "Sad":
        system_mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."
    else:
        system_mode = "You are a helpful AI assistant."

    st.markdown("---")
    st.markdown("**Instructions**:\n- Select a persona.\n- Chat with the AI below.\n- Deployable via Streamlit Community Cloud.")

@st.cache_resource
def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        temperature=0.7,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

try:
    chat_model = get_model()
except Exception as e:
    st.error(f"Error initializing the model. Make sure you have set 'HUGGINGFACEHUB_API_TOKEN' in your environment. Error: {e}")
    st.stop()

if "messages" not in st.session_state or st.session_state.get("current_mode") != system_mode:
    st.session_state.messages = [SystemMessage(content=system_mode)]
    st.session_state.current_mode = system_mode
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    human_msg = HumanMessage(content=prompt)
    st.session_state.display_messages.append(human_msg)
    st.session_state.messages.append(human_msg)
    
    MAX_HISTORY = 10
    if len(st.session_state.messages) > MAX_HISTORY * 2 + 1:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(MAX_HISTORY * 2):]
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat_model.invoke(st.session_state.messages)
                st.markdown(response.content)
                
                ai_msg = AIMessage(content=response.content)
                st.session_state.messages.append(ai_msg)
                st.session_state.display_messages.append(ai_msg)
            except Exception as e:
                st.error(f"Error communicating with the model: {e}")