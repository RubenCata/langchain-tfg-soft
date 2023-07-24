import os
import re
from urllib.parse import urlparse
import streamlit as st
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import vars


#
# --- URL RESPONSE FORMATTER ---
#
def replace_urls_with_fqdn_and_lastpath(text: str):
    # Matches URLs but not thoses like []()
    url_pattern = re.compile(r'''(?<!\])\((?:https?://)\S+\)|(?<!\()\b(?:https?://)\S+\b(?!\))''')

    # Replace all URLs with their FQDN and last path segment
    def replace_url(match):
        url = match.group(0)
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        path_segments = parsed_url.path.split('/')

        if len(path_segments)>3:
            last_segment = path_segments[-1] if path_segments[-1] else path_segments[-2]
            return f"[{scheme}://{netloc}/.../{last_segment}]({url})"
        else:
           return url

    return url_pattern.sub(replace_url, text)


#
# --- WEB FUNCTIONS ---
#
def expander(tab, label, expanded=False, content=""):
    with tab.expander(label=label, expanded=expanded):
        st.write(content)


def get_query():
    question_input = st.chat_input(
        placeholder="Escribe tu pregunta.",
        key="query",
        )
    return question_input


def get_namespace(namespace_options, namespace_selected):
    namespace_input = st.selectbox(
        label="Index Namespace",
        options=namespace_options,
        index=namespace_options.index(namespace_selected),
        key="namespace",
    )
    return namespace_input


#
# --- SESSION STATE ---
#
def check_counter_exist():
    if 'total_in_tokens' not in st.session_state:
        st.session_state.total_in_tokens = 0

    if 'total_out_tokens' not in st.session_state:
        st.session_state.total_out_tokens = 0

    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0

    if 'question_in_tokens' not in st.session_state:
        st.session_state.question_in_tokens = 0

    if 'question_out_tokens' not in st.session_state:
        st.session_state.question_out_tokens = 0

    if 'question_cost' not in st.session_state:
        st.session_state.question_cost = 0


def create_memory():
    # Create a ConversationEntityMemory object if not already created
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="history",
            k=vars.MEMORY_K,
            ai_prefix=vars.AI_NAME,
            human_prefix=vars.username,
            # return_messages=True,
        )


def clear_history():
    if len(st.session_state.memory.chat_memory.messages) > 1:
        del st.session_state.memory
        del st.session_state.sql_conversation_id
        st.session_state.total_in_tokens = 0
        st.session_state.total_out_tokens = 0
        st.session_state.total_cost = 0


def render_history():
    username = os.environ["USERNAME"]
    history=st.session_state.memory.chat_memory.messages
    if len(history) == 0:
        st.session_state.memory.chat_memory.add_ai_message(f"Hola {username}, ¿en qué puedo ayudarte?")

    for m in st.session_state.memory.chat_memory.messages:
        if m.type == "human":
            st.chat_message("user").write(m.content)
        else:
            st.chat_message("assistant").write(m.content)


def get_last_k_history(k) -> List:
    """ Only returns the last K Human/Ai interactions, as List of Human/AI Messages
        suitable to add to the conversation
    """
    history=st.session_state.memory.chat_memory.messages
    return history[-2*k:]