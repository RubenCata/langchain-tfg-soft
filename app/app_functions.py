import os
from uuid import uuid4
import re
import datetime
import hashlib
from urllib.parse import urlparse
import streamlit as st
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
import vars
import sql_alchemy as db
import app_langchain.indexing as indexing


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
    return st.chat_input(
        placeholder="Escribe tu pregunta.",
        key="query",
        )


def get_app_mode(app_mode):
    mode_options = [mode.value for mode in vars.AppMode]
    return st.selectbox(
        label="App Mode",
        options=mode_options,
        index=mode_options.index(app_mode),
        key="app_mode", # Creates a st.session_state.app_mode
        help=f"- __{vars.AppMode.DEFAULT.value}__: Default mode where you can ask questions about internally organized Acciona documentation within different 'Index Namespace'.\n\n- __{vars.AppMode.DOCUMENTS.value}:__ Mode in which you can upload your own documents, ask questions about them, obtain a summary, or compare them with each other."
    )


def get_namespace(namespace_options, namespace_selected, disabled = False):
    return st.selectbox(
        label="Index Namespace",
        options=namespace_options,
        index=namespace_options.index(namespace_selected),
        key="namespace",
        disabled=disabled,
    )


def file_uploader():
    return st.file_uploader(
        label="Upload your document",
        type=['pdf', 'ppt', 'pptx', 'doc', 'docx', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed")


#
# --- PDF FUNCTIONS ---
#
def save_uploaded_docs(index, files, progress_widget):
    path = "../documents"
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

    for file in files:
        data = file.getbuffer()
        with open(path + "/" + file.name,"wb") as f:
            f.write(data)

        md5 = hashlib.md5(data).hexdigest()
        file_extension = os.path.splitext(file.name)[1]
        if file_extension == ".pdf":
            pages = PyMuPDFLoader(path + "/" + file.name).load()
        else:
            pages = UnstructuredFileLoader(path + "/" + file.name, mode="paged", strategy="fast").load()

        os.remove(path + "/" + file.name)

        document_id = str(uuid4())
        chunks = indexing.chunk_doc(pages, file_extension, document_md5=md5)
        if not db.exists_document_md5(md5) and len(chunks) > 0:
            indexing.embed_doc_to_pinecone(index, chunks, progress_widget)
            db.save_document(document_id, file.name, chunks[0]['title'], md5)
            progress_widget.container()
        elif len(chunks) <= 0:
            progress_widget.error('No text has been recognized in the uploaded document.')
        else:
            progress_widget.error(f'The document "{file.name}" is already uploaded under the title of "{db.get_document_title(md5)}"')


#
# --- SESSION STATE HISTORY ---
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
    history=st.session_state.memory.chat_memory.messages[1:]
    return history[-2*k:]


#
# --- CONVERSATIONS TAB FUNCTIONS ---
#
def conversations_display(conversations, container):
    with container:
        now = datetime.datetime.now()
        today_caption = False
        yesterday_caption = False
        week_caption = False
        month_caption = False
        old_caption = False
        for conver in conversations:
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

            if "delete_conversation_id" in st.session_state and conver.id == st.session_state.delete_conversation_id:
                delete_conversation_confirmation_display()
            elif "edit_conversation_id" in st.session_state and conver.id == st.session_state.edit_conversation_id:
                edit_conversation_name_display()

            cols = st.columns((10,5,5))
            with cols[0]:
                conversation_button(conver, button_type)
            with cols[1]:
                edit_conversation_name_button(conver.id, conver.name)
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


def edit_conversation_name_button(conversation_id, conversation_name):
    st.button(":pencil2:", key=str(uuid4()), on_click=confirm_edit_conversation_name, args=(conversation_id, conversation_name, ))

def confirm_edit_conversation_name(conversation_id, conversation_name):
    st.session_state.edit_conversation_id = conversation_id
    st.session_state.edit_conversation_name = conversation_name
    if "delete_conversation_id" and "delete_conversation_name" in st.session_state:
        del st.session_state.delete_conversation_id
        del st.session_state.delete_conversation_name

def cancel_edit_conversation_name():
    del st.session_state.edit_conversation_id
    del st.session_state.edit_conversation_name

def edit_conversation_name_display():
    st.divider()
    st.text_input(
        label="Enter the new name of the conversation:",
        value=st.session_state.edit_conversation_name,
        key="conversation_name",
        on_change=db.edit_conversation_name,
    )
    st.button(label="Cancel", key=str(uuid4()), on_click=cancel_edit_conversation_name, use_container_width=True)
    st.divider()


def del_conversation_button(conversation_id, conversation_name):
    st.button(label=":wastebasket:", key=str(uuid4()), on_click=confirm_delete_conversation, args=(conversation_id, conversation_name, ))

def confirm_delete_conversation(conversation_id, conversation_name):
    st.session_state.delete_conversation_id = conversation_id
    st.session_state.delete_conversation_name = conversation_name
    if "edit_conversation_id" and "edit_conversation_name" in st.session_state:
        del st.session_state.edit_conversation_id
        del st.session_state.edit_conversation_name

def delete_conversation_confirmation_display():
    st.divider()
    st.error('Are you sure you want to delete the conversation "'+ st.session_state.delete_conversation_name[:50] +'"?')
    cols = st.columns((1,1))
    cols[0].button(label="Yes, delete.", key=str(uuid4()), on_click=db.delete_conversation, args=(True, ), type="primary", use_container_width=True)
    cols[1].button(label="No, cancel.", key=str(uuid4()), on_click=db.delete_conversation, args=(False, ), type="primary", use_container_width=True)
    st.divider()


#
# --- DOCUMENTS TAB FUNCTIONS ---
#
def documents_display(index):

    documents = db.get_documents()
    selected = sum(list(map(lambda doc: doc.selected, documents)))
    if selected > 0:
        label = f"{selected} Document{ 's' if selected > 1 else ''} Selected"
    else:
        label = f"Select Documents"

    with st.expander(label=label, expanded=True):
        if "selected_documents" not in st.session_state:
            st.session_state.selected_documents = set()
        for doc in documents:
            if doc.selected:
                st.session_state.selected_documents.add(doc.id)

            if "delete_document_id" in st.session_state and doc.id == st.session_state.delete_document_id:
                delete_document_confirmation_display(index)
            elif "edit_document_id" in st.session_state and doc.id == st.session_state.edit_document_id:
                edit_document_title_display()

            cols = st.columns((10,5,5))
            with cols[0]:
                document_button(doc)
            with cols[1]:
                edit_document_title_button(doc.id, doc.title)
            with cols[2]:
                del_document_button(doc.id, doc.title)


def document_button(doc):
    st.button(
        label=doc.title,
        on_click=db.update_select_doc,
        args=(doc.id, (doc.id not in st.session_state.selected_documents),),
        key=str(uuid4()),
        use_container_width=True,
        type="primary" if doc.selected else "secondary",
    )


def edit_document_title_button(doc_id, doc_title):
    st.button(":pencil2:", key=str(uuid4()), on_click=confirm_edit_document_title, args=(doc_id, doc_title, ))

def confirm_edit_document_title(doc_id, doc_title):
    st.session_state.edit_document_id = doc_id
    st.session_state.edit_document_title = doc_title
    if "delete_document_id" and "delete_document_title" in st.session_state:
        del st.session_state.delete_document_id
        del st.session_state.delete_document_title

def cancel_edit_document_title():
    del st.session_state.edit_document_id
    del st.session_state.edit_document_title

def edit_document_title_display():
    st.divider()
    st.text_input(
        label="Enter the new title of the document:",
        value=st.session_state.edit_document_title,
        key="document_title",
        on_change=db.edit_document_title,
    )
    st.button(label="Cancel", key=str(uuid4()), on_click=cancel_edit_document_title, use_container_width=True)
    st.divider()


def del_document_button(doc_id, doc_title):
    st.button(label=":wastebasket:", key=str(uuid4()), on_click=confirm_delete_document, args=(doc_id, doc_title, ))

def confirm_delete_document(doc_id, doc_title):
    st.session_state.delete_document_id= doc_id
    st.session_state.delete_document_title = doc_title
    if "edit_document_id" and "edit_document_title" in st.session_state:
        del st.session_state.edit_document_id
        del st.session_state.edit_document_title

def delete_document_confirmation_display(index):
    st.divider()
    st.error('Are you sure you want to delete the document "'+ st.session_state.delete_document_title[:50] +'"?')
    cols = st.columns((1,1))
    cols[0].button(label="Yes, delete.", key=str(uuid4()), on_click=db.delete_document, args=(index, True, ), type="primary", use_container_width=True)
    cols[1].button(label="No, cancel.", key=str(uuid4()), on_click=db.delete_document, args=(index, False, ), type="primary", use_container_width=True)
    st.divider()