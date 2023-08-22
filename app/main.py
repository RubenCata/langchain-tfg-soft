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



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- WEB TABS & CONFIG ---
#
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

app_title = "Q&A Deep Learning Bot"
st.set_page_config(
    page_title=app_title,
    initial_sidebar_state='expanded',
)
st.title(app_title)
load_css()

with st.sidebar:
    tab_faq, tab_config, tab_debug = st.tabs(["FAQ", "Config", "Debug"])

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
        st.write("En la pestaña `Config` veras algunas opciones técnicas que puedes modificar, como por ejemplo la fuente de informacion (`Index namespace`).")
        st.write("En la pestaña `Debug` veras algunas trazas y variables internas que podrian ser util para reportar un problema.")

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

chat_model_no_streaming = models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass())



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

app.check_counter_exist()
app.create_memory()

#
# --- TAB 2: CONFIG ---
#
with tab_config:
    slow_down = st.checkbox("Force streaming slowdown", value=True, help="Slowing down streaming of response. Enable for a ChatGPT-like user experience", key="slowdown")

    query_deixis_resolution = st.checkbox("Enable Query Deixis Resolution (Beta)", value=vars.QUERY_DEIXIS_RESOLUTION,
                                          help="Internaly reformulate the user query to avoid ambiguity and ensure that the intended meaning is clear and independent of the conversation context", key="query_deixis_resolution")

    focus_chain_on = st.checkbox("Enable Focus Chain (Beta)", value=vars.FOCUS_CHAIN, help="Focus chain helps the LLM by filtering the documents on metadata", key="focus")

    temp_slider = st.slider('Temperature', value=0.0, min_value=0.0, max_value=1.0, key="temperature")

    min_score = st.slider('Minimal score', 0.70, 0.90, 0.75)

    namespace = app.get_namespace(namespace_options, vars.INDEX_NAMESPACE)


#
# --- MAIN ---
#
app.expander(tab=tab_debug, label="OpenAI API Endpoint", expanded=False, content=os.environ["OPENAI_API_BASE"])

app.render_history()
history = app.get_last_k_history(vars.MEMORY_K)
query = app.get_query()
msg_box = st.empty()

config = {
    "query_deixis_resolution": query_deixis_resolution,
    "focus_chain_on": focus_chain_on,
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

    db.save_conversation(vars.username)


    # Find a metadata filter, only apply to Confluence namespaces
    if focus_chain_on and ("ddo" in namespace):
        response  = chains.get_chat_response(index, config, history, True, query, widgets)
        resp_AIMessage = response["response"]
        filter = response["filter"]

        if filter != None:
            verify = chains.verify_chain(query, resp_AIMessage, chat_model_no_streaming)
            app.expander(tab=tab_debug, label="verify", expanded=False, content=verify )

            if verify.content == "False":
                resp_AIMessage  = chains.get_chat_response(index, config, history, False, query, widgets)["response"]
    else:
        resp_AIMessage  = chains.get_chat_response(index, config, history, False, query, widgets)["response"]

    app.expander(tab=tab_debug, label="resp_AIMessage", expanded=False, content=resp_AIMessage )

    st.session_state.memory.chat_memory.add_ai_message(resp_AIMessage)
    # msg_box.markdown(resp_AIMessage)

    db.save_interaction(query, resp_AIMessage, config)

if "total_cost" in st.session_state and st.session_state.total_cost != 0:
    # st.caption(f"Input tokens: {st.session_state.question_in_tokens}, Output tokens: {st.session_state.question_out_tokens}, Cost: ${st.session_state.question_cost:.2f}")
    # st.caption(f"Total input tokens: {st.session_state.total_in_tokens}, Total output tokens: {st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")

    cols = st.columns((10.9,7.26,2.01,2,2))
    # cols[0].caption(f"Total tokens: {st.session_state.total_in_tokens + st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")
    cols[0].caption(f"Total cost: ${st.session_state.total_cost:.2f}")

    cols[1].button(label=":wastebasket: Eliminar conversación", on_click=app.clear_history, key="del_conversation", use_container_width=True)
    cols[2].button(label=":wastebasket:", on_click=app.clear_history, key="del_conversation_mini", use_container_width=True)
    good_feedback = cols[3].button(
        label=":+1:",
        use_container_width=True
    )
    bad_feedback = cols[4].button(
        label=":-1:",
        use_container_width=True,
    )
    if good_feedback:
        db.save_feedback(True)
    elif bad_feedback:
        db.save_feedback(False)

del st.session_state.question_in_tokens
del st.session_state.question_out_tokens
del st.session_state.question_cost