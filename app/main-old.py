### Imports ###
from enum import Enum
import re
import os
from uuid import UUID
import requests
from urllib.parse import urlparse
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

# import langchain
# langchain.debug = True

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
)
from langchain.callbacks.manager import CallbackManagerForChainRun

from langchain.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import (
    LLMChain,
    TransformChain,
    SequentialChain,
)

from langchain.schema import (
    LLMResult,
)

import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers

from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, ForeignKey, DateTime, UnicodeText, create_engine
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    Session,
)
from datetime import datetime
from uuid import uuid4

import pinecone
import tiktoken

import time
import base64
import json
import random
import yaml



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- SET UP ENVIRONMENT VARIABLES ---
#
config = yaml.safe_load(open("../tfg-config.yml"))

openai_instance=config["openai"]["azure_us"]

os.environ["OPENAI_API_TYPE"] = openai_instance["api_type"]
os.environ["OPENAI_API_KEY"] = openai_instance["api_key"]
os.environ["OPENAI_API_BASE"] = openai_instance["api_base"]
os.environ["OPENAI_API_VERSION"] = openai_instance["api_version"]

pinecone_instance=config["pinecone"]
os.environ["PINECONE_API_KEY"] = pinecone_instance["api_key"]
os.environ["PINECONE_ENVIRONMENT"] = pinecone_instance["environment"]
os.environ["INDEX_NAME"] = "index"
os.environ["INDEX_NAMESPACE"] = "arxiv-11-07-2023"

os.environ["DB_CONNECTION_STR"] = config["db"]
username = config["user"]["username"]

INDEX_NAME = os.environ["INDEX_NAME"]
INDEX_NAMESPACE=os.environ["INDEX_NAMESPACE"]

DEPLOYMENT_NAME = "gpt-35-turbo-v0301"
EMBEDDING_MODEL = "text-embedding-ada-002"
AI_NAME = os.getenv('AI_NAME', 'AI Bot')
FOCUS_CHAIN = (os.getenv('FOCUS_CHAIN', 'True').upper() == "TRUE")
QUERY_DEIXIS_RESOLUTION = (os.getenv('QUERY_DEIXIS_RESOLUTION', 'True').upper() == "TRUE")

MEMORY_K = 5


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
    initial_sidebar_state='collapsed',
)
st.title(app_title)
load_css()


with st.sidebar:
    tab_faq, tab_config, tab_debug = st.tabs(["FAQ", "Config", "Debug"])

with tab_faq:
    with st.expander(label=f"¿Qué es {app_title}?", expanded=False):
        st.write(f"""{app_title} es una aplicacion desarrollada por Rubén Catalán Medina en el trascurso de sus prácticas en la empresa ACCIONA.\n
                 Esta aplicación ha sido adaptada para trabajar en este TFG con una base de conocimiento sobre Deep Learning, de forma que
                 permite hacer preguntas sobre temas relacionados.""")

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



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- TOKEN TYPE ---
#
class TokenType(Enum):
    INPUT = 1
    OUTPUT = 2


#
# --- STREAMLIT SESSION STATE TOKENS ---
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

check_counter_exist()


#
# --- LLM MEMORY ---
#
# Create a ConversationEntityMemory object if not already created
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="history",
        k=MEMORY_K,
        ai_prefix=AI_NAME,
        human_prefix="Human",
        # return_messages=True,
    )


#
# --- CONNECT TO PINECONE ---
#
@st.cache_resource
def get_pinecone_index():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    return pinecone.Index(INDEX_NAME)

# connect to index
index = get_pinecone_index()
namespace_options = sorted(list(index.describe_index_stats()["namespaces"].keys()))


# --- TOKEN COUNT FUNCTIONS ---

def posTratamiento(s: str):
    return s.replace("#OPENED_CURLY_BRACE#", "{").replace("#CLOSED_CURLY_BRACE#", "}")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        posTratamiento(text),
        disallowed_special=()
    )
    return len(tokens)

def add_tokens(text, type: TokenType):
    tokens: float = tiktoken_len(text)
    if type == TokenType.INPUT:
        st.session_state.question_in_tokens += tokens
        st.session_state.total_in_tokens += tokens
    else:
        st.session_state.question_out_tokens += tokens
        st.session_state.total_out_tokens += tokens

    st.session_state.question_cost += 0.002/1000.0 * tokens
    st.session_state.total_cost += 0.002/1000.0 * tokens


#
# --- SQL ALCHEMY ORM BUILD ---
#
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    username = Column(String(100))
    creation_date = Column(DateTime, default=datetime.now)
    interactions = relationship("Interaction", back_populates="conversation", cascade="all, delete", collection_class=list)

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"))
    timestamp = Column(DateTime, default=datetime.now)
    question = Column(UnicodeText)
    response = Column(UnicodeText)
    config = Column(JSON)
    tokens = Column(Integer)
    cost = Column(Float)
    feedback = Column(Boolean)
    conversation = relationship("Conversation", back_populates="interactions")

# Use in local DB only to recreate the DB schema (tables) based on ORM SQLAlchemy definition above
# tmp_engine = create_engine(os.environ["DB_CONNECTION_STR"])
# Base.metadata.create_all(tmp_engine)

def get_sql_session():
    engine = create_engine(
        os.environ["DB_CONNECTION_STR"]
    )

    return Session(engine)

def save_feedback(feedback: bool):
    with get_sql_session() as session:
        session.begin()
        try:
            sql_interaction = session.query(Interaction).filter_by(id=st.session_state.sql_interaction_id).first()
            sql_interaction.feedback = feedback
            session.add(sql_interaction)
            session.commit()
        except:
            session.rollback()
            print("Feedback NO GUARDADO")
            raise


#
# --- AZURE CHAT MODEL ---
#
def get_chat_model(streaming=False, temperature=0.0, handler=None):
    if streaming:
        return AzureChatOpenAI(
            openai_api_base = os.environ["OPENAI_API_BASE"],
            openai_api_version = os.environ["OPENAI_API_VERSION"],
            deployment_name = DEPLOYMENT_NAME,
            openai_api_key = os.environ["OPENAI_API_KEY"],
            openai_api_type = os.environ["OPENAI_API_TYPE"],
            temperature = temperature,
            streaming=streaming,
            callback_manager=BaseCallbackManager([handler]),
            request_timeout=6,
            verbose=True,
        )
    else:
        return AzureChatOpenAI(
            openai_api_base = os.environ["OPENAI_API_BASE"],
            openai_api_version = os.environ["OPENAI_API_VERSION"],
            deployment_name = DEPLOYMENT_NAME,
            openai_api_key = os.environ["OPENAI_API_KEY"],
            openai_api_type = os.environ["OPENAI_API_TYPE"],
            temperature = temperature,
            request_timeout=6,
            verbose=True,
        )


#
# --- FAKE STREAMING CALLBACK ---
#
class FakeStreamingCallbackHandlerClass(BaseCallbackHandler):
    """Fake CallbackHandler."""

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs: Any) -> Any:
        for p in prompts:
            add_tokens(text=p, type=TokenType.INPUT)

    def on_llm_new_token(self, token, **kwargs: Any):
        pass

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id = None, **kwargs: Any) -> Any:
        add_tokens(text=response.generations[0][0].message.content, type=TokenType.OUTPUT)

chat_model_no_streaming = get_chat_model(streaming=True, handler=FakeStreamingCallbackHandlerClass())


#
# --- STREAMING CALLBACK HANDLER ---
#
class MyStreamingCallbackHandlerClass(BaseCallbackHandler):
    """Custom CallbackHandler."""
    widget = None
    incomplete_chat_model_answer: str = ""

    def set_widget(self, c):
        self.widget = c

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs: Any) -> Any:
        self.incomplete_chat_model_answer = ""
        for p in prompts:
            add_tokens(text=p, type=TokenType.INPUT)

    def on_llm_new_token(self, token, **kwargs: Any):
        # Do something with the new token
        self.incomplete_chat_model_answer = self.incomplete_chat_model_answer + token
        self.widget.chat_message("assistant").write( replace_urls_with_fqdn_and_lastpath(self.incomplete_chat_model_answer) )
        if slow_down:
            time.sleep(15/1000)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id = None, **kwargs: Any) -> Any:
        add_tokens(text=response.generations[0][0].message.content, type=TokenType.OUTPUT)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- QUERY TRANSFORM FUNCTION ---
#
def query_transform_func(inputs: dict) -> dict:
    for key in inputs.keys():
        query = inputs[key]
    disambiguation = {
        "CECOC": "CECOC (Centro de Control de Contrucción)",
        "CECOA": "CECOA (Centro de Control de Agua)",
        "CECO": "CECO CECOER (Centro de Control de Energias Renovable)",
        "CECOER": "CECO CECOER (Centro de Control de Energias Renovable)",
    }

    for term in disambiguation.keys():
        query = re.sub(r'\b'+term+r'\b', disambiguation[term], query, flags=re.IGNORECASE)

    # expander(tab=tab_debug, label="query", expanded=False, content=query)
    return {"query": query}


#
# --- DEIXIS RESOLUTION CHAIN ---
#
def create_deixis_resolution_prompt():
    msg_box.chat_message("assistant").write("Rewriting query")
    history_str = st.session_state.memory.load_memory_variables(inputs=[])["history"]
    completion_template = f"""
"Reformula la ultima pregunta de Human en el fragmento siguiente para que sea completa, entendible por cualquiera ajeno a la conversacion y no dependa del contexto anterior de la conversacion.
Tu respuesta debe ser unicamente la pregunta reformulada completa.

<< EJEMPLO >>
Conversacion:
\"\"\"
Human: Que es una red neuronal profunda?
AI Bot:  Una red neuronal profunda es [...] Fuentes: [...]
Human: En que se diferencia de una recurrente?
\"\"\"
Pregunta completa reformulada: ¿En que se diferencia una red neuronal profunda de una recurrente?

<< CONVERSACION ACTUAL >>
\"\"\"
{history_str} \n
""" + """
Human: {query}
\"\"\"

Pregunta completa reformulada:
"""
    return PromptTemplate(template=completion_template, input_variables=["query"])


#
# --- QUERY CLASSIFIER CHAIN ---
#
def query_classifier_chain():
    query_classifier_prompt_template="""Please analyse the following conversation and classify it.
Answer strictly with a valid JSON format, with those fields:

`{{ "Type": $Type, "Name": $Name }}`

IF the conversation sounds related to project management:
THEN ANSWER with:
    Type: "Project"
    Name: Extract the possible name of the project or application object of the project from the sentence

IF the conversation is about an application:
THEN ANSWER with:
    Type: "Application"
    Name: extract the possible name of the application

Otherwise answer with:
    Type: "Other"
    Name: Try to infer a possible subject of sentence (in an IT context)


EXAMPLES:

Conversation: '''Human: Hablame de SCV'''
ANSWER: `{{ "Type": "Application", "Name": "SCV" }}`

[...]

Conversation '''Human: Que es SCV?\n \
    Kate: ...\n
    Human: Y una red neuronal?'''
ANSWER: `{{ "Type": "Application", "Name": "Red neuronal" }}`

Conversation'''
{history}
Human: {query}'''
ANSWER:"""

    return LLMChain(
        llm = chat_model_no_streaming,
        prompt=ChatPromptTemplate.from_template(query_classifier_prompt_template),
        verbose=False,
    )


#
# --- PINECONE CHUNK RETRIEVAL ---
#
def pinecone_chunk_retrieval(inputs, index, namespace, top_k, include_metadata):
    global msg_box
    if "filter" in inputs.keys():
        filter = inputs["filter"]
        inputs.pop("filter")
    else:
        filter = None

    label = ""
    for key in inputs.keys():
        query = inputs[key]
        expander(tab=tab_debug, label=key, expanded=(key=="deixis_query"), content=query)

    msg_box.chat_message("assistant").write("Requesting query embedding")
    query_embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL).embed_query(query)

    msg_box.chat_message("assistant").write("Requesting matching embeddings")
    matching_embeddings = index.query(
        query_embedding,
        top_k=top_k,
        include_metadata=include_metadata,
        namespace=namespace,
        filter=filter,
    )
    return matching_embeddings.matches


class ChunkRetrieval(TransformChain):
    """Custom ChunkRetrieval Chain."""
    index: Any
    namespace: str = ""
    top_k: int = 5
    include_metadata: bool = True

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return {self.output_variables[0]: self.transform(inputs, self.index, self.namespace, self.top_k, self.include_metadata)}


#
# --- FOCUS CHAIN ---
#
def focus_func(inputs: dict, index, namespace, history) -> dict:
    msg_box.chat_message("assistant").write("Filtering results")
    filter=None
    keywords = {
    }
    label = ""
    for key in inputs.keys():
        query = inputs[key]

    output_focus_chain = query_classifier_chain().run({"query": query,"history": history}).replace("`", "")
    expander(tab=tab_debug, label="output_focus_chain", expanded=False, content=output_focus_chain)

    try:
        qclass = json.loads(output_focus_chain)
    except json.JSONDecodeError as e:
        qclass = {}

    expander(tab=tab_debug, label="qclass", expanded=True, content=qclass )

    if "Type" in qclass.keys() and qclass["Type"] in keywords.keys():
        focus_query = keywords[qclass["Type"]] + ", " + qclass["Name"]
        vect = ChunkRetrieval(
            transform=pinecone_chunk_retrieval,
            input_variables=["focus_query"],
            output_variables=["focus_chunk"],
            index=index,
            namespace=namespace,
            top_k=1,
        )({"focus_query": focus_query})["focus_chunk"]
        filter = { "title": vect[0].metadata['title'] }

    expander(tab=tab_debug, label="filter", expanded=True, content=filter )

    return {"filter": filter}


class FocusChain(TransformChain):
    """Custom FocusChain."""
    index: Any
    namespace: str = ""
    history: Any

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return self.transform(inputs, self.index, self.namespace, self.history)


#
# --- CHAT CONVERSATION ---
#
def get_chat_system_template()  -> SystemMessagePromptTemplate:
    system_template = """Te llamas """+AI_NAME+""". Tienes muchos años de experiencia en Deep Learning.
Tu tarea es responder con datos relevantes sobre el tema indicado por el usuario basandote en tus conocimientos.

En particular, encontraste estos fragmentos de informacion que podrían ser relevantes: \"\"\"\n
{formatted_chunks}
\n\"\"\"

Sobre tu respuesta:
- Responde al usuario usando al máximo toda la información que conoces.
- Sé lo más detallada posible.
- Si no puedes encontrar la respuesta en lo que conoces, responde honestamente "No lo sé", haciendo un resumen rápido de la información y pidiéndole al usuario que reescriba su consulta.
- Responde en español.
- Responde en formato bullet list o tabla cuando aplique.
- Al final de tu respuesta, escribe la lista de Fuentes [Título](URL) de los fragmentos que utilizaste para tu respuesta.
"""
# Glosario:
# -

    # Randomly address the user by its name
    if username != "" and random.randint(0,10) == 0:
        system_template += f"\n\nRecuerda llamar al usuario por su nombre: {username}."

    return SystemMessagePromptTemplate.from_template(template=system_template)

def create_chat_conversation(history):
    conversation = [
            get_chat_system_template(),
        ]
    if history != None:
        conversation = conversation + history[1:]
    conversation.append(HumanMessagePromptTemplate.from_template(template="{query}"))

    expander(tab=tab_debug, label="conversation_template", expanded=False, content=conversation)
    return conversation


#
# --- CHUNK FORMATTER ---
#
class ChunkFormatter(TransformChain):
    """Custom Chunk Formatter Transform Chain."""
    input_variables: List[str] = ["chunks"]
    output_variables: List[str] = ["formatted_chunks"]

    # --- PRE y POS TRATAMIENTO ---
    def _preTratamiento(self, s: str):
        return s.replace("{", "#OPENED_CURLY_BRACE#").replace("}", "#CLOSED_CURLY_BRACE#")

    def _posTratamiento(self, s: str):
        return s.replace("#OPENED_CURLY_BRACE#", "{").replace("#CLOSED_CURLY_BRACE#", "}")


    # --- TOKEN COUNT FUNCTION ---
    def _tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        # tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        tokens = tokenizer.encode(
            self._posTratamiento(text),
            disallowed_special=()
        )
        return len(tokens)


    # --- FORMAT CHUNKS AS EXAMPLES ---
    def _format_chunks(self, chunks):
        global msg_box
        msg_box.chat_message("assistant").write("Creating Prompt")

        # Return a array [] of dicts {"chunk": chunk} needed for the example selector
        formatted_chunks = [
            f"Fuente: [{item['metadata']['title']}]({item['metadata']['url']})"
            f"\n\n"
            f"Relevancia: {100*item['score']:.2f}%"
            f"\n\n"
            f"Título: {item['metadata']['text']}"

            for item in chunks if item['score'] > min_score
        ]
        return [{"chunk": self._preTratamiento(chunk)} for chunk in formatted_chunks]


    # --- FEW SHOT EXAMPLE SELECTOR ---
    def _few_shot_chunk_selector(self, formatted_chunks):
        chunk_prompt_template = PromptTemplate(
            input_variables=["chunk"],
            template="{chunk}",
        )

        chunk_selector = LengthBasedExampleSelector(
            examples=formatted_chunks,
            example_prompt=chunk_prompt_template,
            max_length=3500,
            get_text_length=self._tiktoken_len,
        )

        formatted_chunks_selection = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=chunk_selector,
            example_prompt=chunk_prompt_template,
            prefix='',
            suffix='',
            input_variables=[],
            example_separator="\n\n---\n\n"
        ).format()
        return self._posTratamiento(formatted_chunks_selection)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        formatted_chunks = self._format_chunks(inputs["chunks"])
        formatted_chunks = self._few_shot_chunk_selector(formatted_chunks)
        expander(tab=tab_debug, label="formatted_chunks", expanded=False, content=formatted_chunks)
        return {self.output_variables[0]: formatted_chunks}


#
# --- VERIFY CHAIN ---
#
def verify_chain(query, response, chat_model: None):
    verify_template = """
For the following given question and response, identify if the response answers to the same topic asked in the question or if it needs more information.

Question: {query}

Response: {response}

The output must be a boolean. \
Answer False if the response says that it doesnt have enough information or if it asks the user for more information or is an incompleted answer\
Answer True otherwise.
    """
    verify_messages = ChatPromptTemplate.from_template(template=verify_template).format_messages(query=query, response=response)
    return chat_model(verify_messages)


#
# --- SEARCH SEQUENTIAL CHAIN ---
#
def create_search_sequential_chain(input_variables, index, namespace, history, llm):
    chains = []
    chains.append(ChunkRetrieval(
        transform=pinecone_chunk_retrieval,
        input_variables=["deixis_query", "filter"] if "filter" in input_variables else ["deixis_query"],
        output_variables= ["chunks"],
        index=index,
        namespace=namespace,
    ))
    chains.append(ChunkFormatter(
        transform=dummy_func,
        input_variables=["chunks"],
        output_variables=["formatted_chunks"],
    ))

    conversation = create_chat_conversation(history)
    chains.append(LLMChain(
            llm = llm,
            prompt=ChatPromptTemplate.from_messages(conversation), #Takes "query"
            output_key="response",
        ))

    return SequentialChain(
        chains=chains,
        input_variables=input_variables,
        output_variables=["response"],
        # verbose=True,
    )


#
# --- GENERIC SEQUENTIAL CHAIN ---
#
def create_sequential_chain(llm, index, namespace, history, focus_chain_on):
    chains = []

    chains.append(TransformChain(
        transform=query_transform_func,
        input_variables=["initial_query"],
        output_variables=["query"],
    ))
    if query_deixis_resolution:
        chains.append(LLMChain(
            llm=chat_model_no_streaming,
            prompt=create_deixis_resolution_prompt(),
            output_key="deixis_query",
        ))
    else:
        chains.append(TransformChain(
            transform=dummy_func,
            input_variables=["query"],
            output_variables=["deixis_query"],
        ))
    # Find a metadata filter, only apply to Confluence namespaces
    if focus_chain_on and ("ddo" in namespace or "rub" in namespace):
        output_variables = ["response", "filter"]
        chains.append(FocusChain(
            transform=focus_func,
            input_variables=["deixis_query"],
            output_variables=["filter"],
            index=index,
            namespace=namespace,
            history=history,
        ))
        chains.append(create_search_sequential_chain(
            input_variables=["query", "deixis_query", "filter"],
            index=index,
            namespace=namespace,
            history=history,
            llm= llm,
        ))
    else:
        output_variables=["response"]
        chains.append(create_search_sequential_chain(
            input_variables=["query", "deixis_query"],
            index=index,
            namespace=namespace,
            history=history,
            llm= llm,
        ))

    return SequentialChain(
        chains=chains,
        input_variables=["initial_query"],
        output_variables=output_variables,
        # verbose=True,
    )


#
# --- SEQUENTIAL CHAIN RESPONSE ---
#
def get_chat_response(llm, index, namespace, history, focus_chain_on, query):
    return create_sequential_chain(
        llm=llm,
        index=index,
        namespace=namespace,
        history=history,
        focus_chain_on=focus_chain_on,
    )({"initial_query":query})


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



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- WEB FUNCTIONS ---
#
def expander(tab, label, expanded=False, content=""):
    with tab.expander(label=label, expanded=expanded):
        st.write(content)

def get_query():
    question_input = st.chat_input(placeholder="Escribe tu pregunta.", key="query")
    return question_input

def get_namespace():
    namespace_input = st.selectbox("Index Namespace", namespace_options, index=namespace_options.index(INDEX_NAMESPACE), key="namespace")
    return namespace_input

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
        st.session_state.memory.chat_memory.add_ai_message(f"Hola {username}, ¿en qué puedo ayudarte?")

    for m in st.session_state.memory.chat_memory.messages:
        if m.type == "human":
            st.chat_message("user").write(m.content)
        else:
            st.chat_message("assistant").write(m.content)

def get_last_k_history() -> List:
    """ Only returns the last K Human/Ai interactions, as List of Human/AI Messages
        suitable to add to the conversation
    """
    history=st.session_state.memory.chat_memory.messages
    return history[ -2 * MEMORY_K : ]

def dummy_func(inputs: dict) -> dict:
    return inputs


#
# --- TAB 2: CONFIG ---
#
with tab_config:
    slow_down = st.checkbox("Force streaming slowdown", value=True, help="Slowing down streaming of response. Enable for a ChatGPT-like user experience", key="slowdown")

    query_deixis_resolution = st.checkbox("Enable Query Deixis Resolution (Beta)", value=QUERY_DEIXIS_RESOLUTION,
                                          help="Internaly reformulate the user query to avoid ambiguity and ensure that the intended meaning is clear and independent of the conversation context", key="query_deixis_resolution")

    focus_chain_on = st.checkbox("Enable Focus Chain (Beta)", value=FOCUS_CHAIN, help="Focus chain helps the LLM by filtering the documents on metadata", key="focus")

    temp_slider = st.slider('Temperature', value=0.0, min_value=0.0, max_value=1.0, key="temperature")

    min_score = st.slider('Minimal score', 0.70, 0.90, 0.75)

    namespace = get_namespace()



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



#
# --- LLM CHAT MODEL ---
#
MyStreamingCallbackHandler = MyStreamingCallbackHandlerClass()
chat_model = get_chat_model(streaming=True, temperature=temp_slider, handler=MyStreamingCallbackHandler)


#
# --- MAIN ---
#
expander(tab=tab_debug, label="OpenAI API Endpoint", expanded=False, content=os.environ["OPENAI_API_BASE"])

st.empty()
render_history()
history = get_last_k_history()
query = get_query()
msg_box = st.empty()

if query:
    st.chat_message("user").write(query)
    st.session_state.memory.chat_memory.add_user_message(query)

    msg_box = st.empty()
    # output_box = msg_box.chat_message("assistant")

    # Reset MyStreamingCallbackHandler instance
    MyStreamingCallbackHandler.set_widget(msg_box)

    if 'sql_conversation_id' not in st.session_state:
        with get_sql_session() as session:
            session.begin()
            sql_conversation = Conversation(id=str(uuid4()), username=username)
            try:
                session.add(sql_conversation)
                session.commit()
            except:
                session.rollback()
                raise
            else:
                st.session_state.sql_conversation_id = sql_conversation.id


    # Find a metadata filter, only apply to Confluence namespaces
    if focus_chain_on:
        response  = get_chat_response(chat_model, index, namespace, history, True, query)
        resp_AIMessage = response["response"]
        filter = response["filter"]

        if filter != None:
            verify = verify_chain(query, resp_AIMessage, chat_model_no_streaming)
            expander(tab=tab_debug, label="verify", expanded=False, content=verify )

            if verify.content == "False":
                # Fake Streaming...
                for word in ["\n\n", "Voy", " ", "a", " ", "buscar", " ", "una", " ", "mejor", " ", "respuesta", "..."]:
                    MyStreamingCallbackHandler.on_llm_new_token(word)
                time.sleep(1)

                resp_AIMessage  = get_chat_response(chat_model, index, namespace, history, False, query)["response"]
    else:
        resp_AIMessage  = get_chat_response(chat_model, index, namespace, history, False, query)["response"]

    expander(tab=tab_debug, label="resp_AIMessage", expanded=False, content=resp_AIMessage )

    st.session_state.memory.chat_memory.add_ai_message(replace_urls_with_fqdn_and_lastpath(resp_AIMessage))
    # msg_box.markdown(replace_urls_with_fqdn_and_lastpath(resp_AIMessage))

    config = {
        "query_deixis_resolution": query_deixis_resolution,
        "focus_chain_on": focus_chain_on,
        "temp_slider": temp_slider,
        "min_score": min_score,
        "namespace": namespace,
    }

    with get_sql_session() as session:
        session.begin()
        interaction = Interaction(
                id=str(uuid4()),
                question=query,
                response=replace_urls_with_fqdn_and_lastpath(resp_AIMessage),
                config=config,
                tokens=st.session_state.question_in_tokens + st.session_state.question_out_tokens,
                cost=st.session_state.question_cost,
            )
        st.session_state.sql_interaction_id = interaction.id
        try:
            sql_conversation = session.query(Conversation).filter_by(id=st.session_state.sql_conversation_id).first()

            sql_conversation.interactions.append(interaction)
            session.add(sql_conversation)
            session.commit()
        except:
            session.rollback()
            print("Interaccion NO GUARDADA", interaction.id)
            raise

    del st.session_state.question_in_tokens
    del st.session_state.question_out_tokens
    del st.session_state.question_cost

if "total_cost" in st.session_state and st.session_state.total_cost != 0:
    # st.caption(f"Input tokens: {st.session_state.question_in_tokens}, Output tokens: {st.session_state.question_out_tokens}, Cost: ${st.session_state.question_cost:.2f}")
    # st.caption(f"Total input tokens: {st.session_state.total_in_tokens}, Total output tokens: {st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")

    cols = st.columns((10.9,7.26,2.01,2,2))
    # cols[0].caption(f"Total tokens: {st.session_state.total_in_tokens + st.session_state.total_out_tokens}, Total cost: ${st.session_state.total_cost:.2f}")
    cols[0].caption(f"Total cost: ${st.session_state.total_cost:.2f}")

    cols[1].button(label=":wastebasket: Eliminar conversación", on_click=clear_history, key="del_conversation", use_container_width=True)
    cols[2].button(label=":wastebasket:", on_click=clear_history, key="del_conversation_mini", use_container_width=True)
    good_feedback = cols[3].button(
        label=":+1:",
        use_container_width=True
    )
    bad_feedback = cols[4].button(
        label=":-1:",
        use_container_width=True,
    )
    if good_feedback:
        save_feedback(True)
    elif bad_feedback:
        save_feedback(False)