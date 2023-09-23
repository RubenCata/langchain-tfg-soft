### Imports ###
import os
import time

# import langchain
# langchain.debug = True

import streamlit as st
import pinecone

import vars
import sql_alchemy as db
import app_functions as app

import app_langchain.models as models
import app_langchain.chains as chains
import app_langchain.tokens as tokens



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- WEB TABS & CONFIG ---
#
def load_css():
    with open("app/static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

app_title = "Q&A Deep Learning Bot"
st.set_page_config(
    page_title=app_title,
    initial_sidebar_state='expanded',
)
st.title(app_title)
load_css()

if "app_mode" not in st.session_state:
    app_mode = vars.AppMode.DEFAULT.value
else:
    app_mode = st.session_state.app_mode

with st.sidebar:
    app_mode = app.get_app_mode(app_mode)

    if app_mode == vars.AppMode.DEFAULT.value:
        tab_conversations, tab_config, tab_faq, tab_debug = st.tabs(["Conversations", "Config", "FAQ", "Debug"])
    else:
       tab_conversations, tab_docs, tab_config, tab_faq, tab_debug = st.tabs(["Conversations", "Documents", "Config", "FAQ", "Debug"])

with tab_faq:
    with st.expander(label=f"¿Qué es {app_title}?", expanded=False):
        st.write(f"{app_title} es una aplicacion desarrollada por Rubén Catalán Medina en el trascurso de sus prácticas en la empresa ACCIONA.")
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

    with st.expander(label=f"Modo {vars.AppMode.DOCUMENTS.value}", expanded=False):
        st.write(f'En la pestaña de `Config` también verás que puedes cambiar el modo de la aplicación a "{vars.AppMode.DOCUMENTS.value}".')
        st.write("En este modo, se te mostrará una nueva pestaña `Documents` en la que puedes subir tus propios documentos y hacer preguntas sobre uno o varios de ellos a la vez.")

    st.write("## Disclaimer")
    st.caption("""Esta aplicación genera las respuestas basandose en fragmentos de texto, extraidos de artículos académicos. \
        Aun así, OpsGPT genera las respuestas usando un modelo de lenguaje natural (es decir, usa inteligencia artificial), y puede equivocarse. \
        Revisa bien las respuestas y las fuentes proporcionadas en ellas para asegurate de que sean correctas.""")

#
# --- CREATE MYSQL DATABASE ---
#
# Use in local DB only to recreate the DB schema (tables) based on ORM SQLAlchemy definition
# db.create_database()



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- CONNECT TO PINECONE ---
#
@st.cache_resource
def get_pinecone_index():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    return pinecone.Index(vars.INDEX_NAME)

# connect to index
index = get_pinecone_index()
namespace_options = sorted(list(index.describe_index_stats()["namespaces"].keys()))



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



tokens.check_counter_exist()
app.create_memory()


#
# --- TAB CONVERSATIONS ---
#
with tab_conversations:
    st.button(":heavy_plus_sign: New Conversation", on_click=app.clear_history, use_container_width=True)
    st.divider()
    db.get_user_conversations(st.container())


#
# --- TAB CONFIG ---
#
with tab_config:
    slow_down = st.checkbox("Force streaming slowdown", value=True, help="Slowing down streaming of response. Enable for a ChatGPT-like user experience", key="slowdown")

    query_deixis_resolution = st.checkbox("Enable Query Deixis Resolution (Beta)", value=vars.QUERY_DEIXIS_RESOLUTION,
                                          help="Internaly reformulate the user query to avoid ambiguity and ensure that the intended meaning is clear and independent of the conversation context", key="query_deixis_resolution")


    temp_slider = st.slider('Temperature', value=0.0, min_value=0.0, max_value=1.0, key="temperature",
                            help="How creative do you want the AI to be. A value close to 0 will be more precise, while a value close to 1 will be more creative.")

    min_score = st.slider('Minimal score', 0.70, 0.90, 0.75)

    if st.session_state.app_mode == vars.AppMode.DEFAULT.value:
        namespace_options.pop(namespace_options.index("uploaded-documents"))
        namespace = app.get_namespace(namespace_options, vars.INDEX_NAMESPACE)
    else:
        namespace = app.get_namespace(namespace_options, "uploaded-documents", disabled = True)


#
# --- TAB DOCUMENTS ---
#
if app_mode == vars.AppMode.DOCUMENTS.value:
    with tab_docs:
        with tab_docs.expander(label="Upload Document", expanded=False):
            with st.form(key="upload-pdf-form", clear_on_submit=True):
                files = app.file_uploader()
                progress_widget = st.empty()
                submitted = st.form_submit_button("Upload Documents", type="primary", use_container_width=True)

            if submitted and files:
                app.save_uploaded_docs(index, files, progress_widget)

        with tab_docs.expander(label="Select Documents", expanded=True):
            db.get_user_documents(index, st.container())

#
# --- MAIN ---
#
app.expander(tab=tab_debug, label="OpenAI API Endpoint", expanded=False, content=os.environ["OPENAI_API_BASE"])

app.render_history()
history = app.get_last_k_history(vars.MEMORY_K)
query = app.get_query()
msg_box = st.empty()

config = {
    "app_mode": app_mode,
    "query_deixis_resolution": query_deixis_resolution,
    "temp_slider": temp_slider,
    "min_score": min_score,
    "namespace": namespace,
}

if query:
    st.chat_message("user").write(query)
    st.session_state.memory.chat_memory.add_user_message(query)

    msg_box = st.empty()
    # output_box = msg_box.chat_message("assistant")

    # Reset MyStreamingCallbackHandler instance
    MyStreamingCallbackHandler = models.MyStreamingCallbackHandlerClass()
    MyStreamingCallbackHandler.set_slow_down(slow_down)
    MyStreamingCallbackHandler.set_widget(msg_box)

    widgets = {
        "tab_debug": tab_debug,
        "msg_box": msg_box,
        "slow_down": slow_down,
    }

    db.save_conversation()


    if app_mode == vars.AppMode.DEFAULT.value:
        response = chains.get_chat_response(index, config, history, query, widgets=widgets)

    elif app_mode == vars.AppMode.DOCUMENTS.value:
        documents = db.get_selected_documents()
        response = chains.get_chat_response(index, config, history, query, widgets=widgets, documents=documents)

    ai_feedback = response["ai_feedback"]
    try:
        if ai_feedback != None:
            ai_feedback = bool(ai_feedback)
    except:
        ai_feedback = False
    app.expander(tab=tab_debug, label="ai_feedback", expanded=False, content=ai_feedback)

    resp_AIMessage = response["response"]
    db.save_interaction(query, resp_AIMessage, config, ai_feedback, response["chunks"], response["deixis_query"])

    app.expander(tab=tab_debug, label="resp_AIMessage", expanded=False, content=resp_AIMessage)
    st.session_state.memory.chat_memory.add_ai_message(resp_AIMessage)

if "total_cost" in st.session_state and st.session_state.total_cost != 0:
    # st.caption(f"Input tokens: {st.session_state.question_in_tokens}, Output tokens: {st.session_state.question_out_tokens}, Cost: ${st.session_state.question_cost:.2f}")
    # st.caption(f"Total input tokens: {st.session_state.total_in_tokens}, Total output tokens: {st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")

    cols = st.container().columns((10,5,5))
    # cols[0].caption(f"Total tokens: {st.session_state.total_in_tokens + st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")
    cols[0].caption(f"Total cost: ${st.session_state.total_cost:.2f}")
    good_feedback = cols[1].button(label=":+1:", use_container_width=True)
    bad_feedback = cols[2].button(label=":-1:", use_container_width=True)
    if good_feedback:
        db.save_feedback(True)
    elif bad_feedback:
        db.save_feedback(False)

del st.session_state.question_in_tokens
del st.session_state.question_out_tokens
del st.session_state.question_cost