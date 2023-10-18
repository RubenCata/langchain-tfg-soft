### Imports ###
import streamlit as st
app_title = "Q&A Deep Learning Bot"
st.set_page_config(
    page_title=app_title,
    initial_sidebar_state='expanded',
)

import app_functions as app
import app_langchain.chains as chains



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


#
# --- CREATE MYSQL DATABASE ---
#
# Use in local DB only to recreate the DB schema (tables) based on ORM SQLAlchemy definition
# app.db.create_database()


#
# --- WEB TABS & CONFIG ---
#
def load_css():
    with open("app/static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

load_css()

if "app_mode" not in st.session_state:
    app_mode = app.vars.AppMode.DEFAULT.value
else:
    app_mode = st.session_state.app_mode

with st.sidebar:
    app_mode = app.get_app_mode(app_mode)

    if app_mode == app.vars.AppMode.DEFAULT.value:
        main_title = "Q&A Deep Learning"
        tab_conversations, tab_config, tab_faq, tab_debug = st.tabs(["Conversations", "Config", "FAQ", "Debug"])
    else:
       main_title = "Q&A Private Documents"
       tab_conversations, tab_docs, tab_config, tab_faq, tab_debug = st.tabs(["Conversations", "Documents", "Config", "FAQ", "Debug"])

st.title(main_title)
app.vars.connect_to_pinecone()
app.init_memory()

#
# --- FAQ ---
#
with tab_faq:
    with st.expander(label=f"¿Qué es esta aplicación?", expanded=False):
        st.write(f"Es una aplicacion desarrollada por Rubén Catalán Medina en el trascurso de sus prácticas en la empresa ACCIONA.")
        st.write("Esta aplicación ha sido adaptada para trabajar en este TFG con una base de conocimiento sobre Deep Learning, de forma que permite hacer preguntas sobre temas relacionados.")

    with st.expander(label=f"¿Qué documentación se ha usado para crear la base de conocimento?", expanded=False):
        st.write(f"""Para la base de conocimiento centrada en Deep Learning se han usado +3500 documentos sobre Machine Learning e Inteligencia Artificial \
            centrados en Deep Learning procedentes de arXiv. arXiv es un servicio de distribución gratuito y un archivo de acceso abierto con más de \
            2 millones de artículos académicos sobre distintos campos científicos entre los que están la ciencia de la computación y matemáticas.""")

    with st.expander(label="Como usarla?", expanded=False):
        st.write("Escribe tu pregunta, pulsa Intro y se generará la respuesta.")
        st.write("En la pestaña `Conversations` se guardarán las distintas conversaciones que tengas en la aplicación para que puedas continuarlas en el tiempo.")
        st.write("En la pestaña `Config` verás algunas opciones técnicas que puedes modificar, como, por ejemplo, la fuente de información (`Index Namespace`).")
        st.write("En la pestaña `Debug` verás algunas trazas y variables internas que podrían ser útiles para reportar un problema.")

    with st.expander(label=f"Modo {app.vars.AppMode.DOCUMENTS.value}", expanded=False):
        st.write(f'En la pestaña de `Config` también verás que puedes cambiar el modo de la aplicación a "{app.vars.AppMode.DOCUMENTS.value}".')
        st.write("En este modo, se te mostrará una nueva pestaña `Documents` en la que puedes subir tus propios documentos y hacer preguntas sobre uno o varios de ellos a la vez.")

    st.write("## Disclaimer")
    st.caption("""Esta aplicación genera las respuestas basandose en fragmentos de texto, extraidos de artículos académicos. \
        Aun así, la aplicación genera las respuestas usando un modelo de lenguaje natural (es decir, usa inteligencia artificial), y puede equivocarse. \
        Revisa bien las respuestas y las fuentes proporcionadas en ellas para asegurate de que sean correctas.""")


#
# --- TAB CONFIG ---
#
with tab_config:
    slow_down = st.checkbox("Force streaming slowdown", value=True, help="Slowing down streaming of response. Enable for a ChatGPT-like user experience", key="slowdown")

    query_deixis_resolution = st.checkbox("Enable Query Deixis Resolution (Beta)", value=app.vars.QUERY_DEIXIS_RESOLUTION,
                                          help="Internaly reformulate the user query to avoid ambiguity and ensure that the intended meaning is clear and independent of the conversation context", key="query_deixis_resolution")

    temp_slider = st.slider('Temperature', value=0.0, min_value=0.0, max_value=1.0, key="temperature",
                            help="How creative do you want the AI to be. A value close to 0 will be more precise, while a value close to 1 will be more creative.")

    min_score = st.slider('Minimal score', 0.70, 0.90, 0.75)

    if st.session_state.app_mode == app.vars.AppMode.DEFAULT.value:
        for docs_namespace in [s for s in app.vars.namespace_options if 'documents' in s]:
            app.vars.namespace_options.pop(app.vars.namespace_options.index(docs_namespace))
        namespace = app.get_namespace(app.vars.namespace_options, app.vars.INDEX_NAMESPACE)
    else:
        if "uploaded-documents" not in app.vars.namespace_options:
            app.indexing.inicialize_doc_namespace(app.vars.index, "uploaded-documents")
            app.vars.namespace_options.append("uploaded-documents")
        namespace = app.get_namespace(app.vars.namespace_options, "uploaded-documents", disabled = True)

    config = {
        "app_mode": app_mode,
        "query_deixis_resolution": query_deixis_resolution,
        "temp_slider": temp_slider,
        "min_score": min_score,
        "namespace": namespace,
    }


#
# --- TAB CONVERSATIONS ---
#
with tab_conversations:
    st.button(":heavy_plus_sign: New Conversation", on_click=app.clear_history, use_container_width=True)
    st.divider()
    app.conversations_display(st.container())


#
# --- TAB DOCUMENTS ---
#
if app_mode == app.vars.AppMode.DOCUMENTS.value:
    with tab_docs:
        with tab_docs.expander(label="Upload Documents", expanded=False):
            with st.form(key="upload-pdf-form", clear_on_submit=True):
                files = app.file_uploader()
                progress_widget = st.empty()
                submitted = st.form_submit_button("Upload Documents", type="primary", use_container_width=True)

            if submitted and files:
                app.save_uploaded_docs(files, progress_widget)

        app.documents_display()


#
# --- MAIN ---
#
history = app.get_last_k_history(app.vars.MEMORY_K)
query = app.get_query()
msg_box = st.empty()

if query:
    st.chat_message("user").write(query)
    st.session_state.memory.chat_memory.add_user_message(query)

    msg_box = st.empty()

    widgets = {
        "tab_debug": tab_debug,
        "msg_box": msg_box,
        "slow_down": slow_down,
    }

    app.create_conversation()


    if app_mode == app.vars.AppMode.DEFAULT.value:
        response = chains.get_chat_response(app.vars.index, config, history, query, widgets=widgets)

    elif app_mode == app.vars.AppMode.DOCUMENTS.value:
        documents = app.db.get_selected_documents()
        response = chains.get_chat_response(app.vars.index, config, history, query, widgets=widgets, documents=documents)

    ai_feedback = response["ai_feedback"]
    try:
        if ai_feedback != None:
            ai_feedback = bool(ai_feedback)
    except:
        ai_feedback = False
    chains.expander(tab=tab_debug, label="ai_feedback", expanded=False, content=ai_feedback)

    resp_AIMessage = response["response"]

    conver_name = app.db.get_conversation(st.session_state.sql_conversation_id).name
    if conver_name is None:
        conver_name = chains.naming_chain(query, response).content
    app.db.save_interaction(conver_name, query, resp_AIMessage, config, ai_feedback, response["chunks"], response["deixis_query"])

    chains.expander(tab=tab_debug, label="resp_AIMessage", expanded=False, content=resp_AIMessage)
    st.session_state.memory.chat_memory.add_ai_message(resp_AIMessage)


app.tokens_and_feedback_display()
del st.session_state.question_in_tokens
del st.session_state.question_out_tokens
del st.session_state.question_cost