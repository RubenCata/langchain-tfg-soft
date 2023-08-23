import os
from uuid import uuid4
import re
import datetime
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
import sql_alchemy as db


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
def create_memory(recreate: bool = False):
    # Create a ConversationEntityMemory object if not already created
    if 'memory' not in st.session_state or recreate:
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
    history=st.session_state.memory.chat_memory.messages
    if len(history) == 0:
        st.session_state.memory.chat_memory.add_ai_message(f"Hola {vars.username}, ¿en qué puedo ayudarte?")
        # When we load previous conversations, we add this msg manually. Modify it also there if needed.

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


def user_conversations_display(user_conversations, container):
    with container:
        now = datetime.datetime.now()
        today_caption = False
        yesterday_caption = False
        week_caption = False
        month_caption = False
        old_caption = False
        for conver in user_conversations:
            if "sql_conversation_id" in st.session_state and conver.id == st.session_state.sql_conversation_id:
                button_type = "primary"
            else:
                button_type = "secondary"

            dif = (now - conver.interactions[0].timestamp).days
            if not today_caption and dif < 1:
                st.caption("Today")
                today_caption = True
            elif not yesterday_caption and 1 <= dif < 2:
                st.caption("Yesterday")
                yesterday_caption = True
            elif not week_caption and 2 <= dif < 7:
                st.caption("Last week")
                week_caption = True
            elif not month_caption and 7 <= dif <= 30:
                st.caption("Last month")
                month_caption = True
            elif not old_caption and dif > 30:
                st.caption("Old")
                old_caption = True

            cols = st.columns((10,5,5))
            with cols[0]:
                conversation_button(conver, button_type)
            with cols[1]:
                edit_name_button(conver.id, conver.name)
            with cols[2]:
                del_conversation_button(conver.id, conver.name)


def conversation_button(conversation, button_type):
    if conversation.name == None:
        label = "New Conversation"
    else:
        label = conversation.name
    st.button(
        label=label,
        on_click=db.load_conversation,
        args=(conversation.id,),
        key=str(uuid4()),
        use_container_width=True,
        type=button_type,
    )


def edit_name_button(conversation_id, conversation_name):
    st.button(":pencil2:", key=str(uuid4()), on_click=confirm_edit_name, args=(conversation_id, conversation_name, ))

def confirm_edit_name(conversation_id, conversation_name):
    st.session_state.edit_conversation_id = conversation_id
    st.session_state.edit_conversation_name = conversation_name

def cancel_edit_name():
    del st.session_state.edit_conversation_id
    del st.session_state.edit_conversation_name

def edit_conversation_name_display(widget):
    widget.divider()
    new_conversation_name = widget.text_input(label="Enter the new name of the conversation:", placeholder=st.session_state.edit_conversation_name, key="conversation_name")
    cols = widget.columns((1,1))
    cols[0].button(label="Submit", on_click=db.edit_conversation_name, args=(new_conversation_name,), type="primary", use_container_width=True)
    cols[1].button(label="Cancel", on_click=cancel_edit_name, use_container_width=True)
    widget.divider()


def del_conversation_button(conversation_id, conversation_name):
    st.button(label=":wastebasket:", key=str(uuid4()), on_click=confirm_delete, args=(conversation_id, conversation_name, ))

def confirm_delete(conversation_id, conversation_name):
    st.session_state.delete_conversation_id = conversation_id
    st.session_state.delete_conversation_name = conversation_name

def delete_confirmation_display(widget):
    widget.divider()
    widget.error('Are you sure you want to delete the conversation "'+ st.session_state.delete_conversation_name[:50] +'"?')
    cols = widget.columns((1,1))
    cols[0].button(label="Yes, delete.", on_click=db.delete_conversation, args=(True, ), type="primary", use_container_width=True)
    cols[1].button(label="No, cancel.", on_click=db.delete_conversation, args=(False, ), type="primary", use_container_width=True)
    widget.divider()