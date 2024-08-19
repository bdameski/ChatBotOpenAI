import streamlit as st
from repository.graph_db import check_graph_db_connection
from service.agent import agent_executor
import os
import dotenv

dotenv.load_dotenv()

st.title('🦜🔗 News Chatbot')

openai_api_key = os.getenv('OPENAI_API_KEY')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(st.session_state.messages)

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream_output = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})["output"]
        print("STREAM: ", stream_output)
        st.markdown(stream_output)
        # response = st.write_stream(stream)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": stream_output})
