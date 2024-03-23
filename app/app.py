"""
This module serves as entrypoint to this RAG application.
References: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#write-the-app
"""
import streamlit as st
from rag import RAGController


controller = RAGController()

st.title("METCS777 - Demo")
topic = st.radio(label="Document Topic", 
               options=["characters", "region"],
               key="topic")

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter question here?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = controller.run(query=prompt, search_kwargs={"filter": {"topic":topic}})
        response = st.write(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": stream})