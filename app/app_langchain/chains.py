import json
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

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

from langchain.chains import (
    LLMChain,
    TransformChain,
    SequentialChain,
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManagerForChainRun

import tiktoken
import random
import streamlit as st
import vars
import app_functions as app
import sql_alchemy as db

import app_langchain.models as models


def dummy_func(inputs: dict) -> dict:
    return inputs


#
# --- DUMMY FUNCTION ---
#
class DummyChain(TransformChain):
    """Custom Dummy Chain."""
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return {self.output_variables[0]: inputs[self.input_variables[0]]}


#
# --- DEIXIS RESOLUTION CHAIN ---
#
def create_deixis_resolution_prompt(widgets):
    widgets['msg_box'].chat_message("assistant").write("Rewriting query")
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
# --- PINECONE CHUNK RETRIEVAL ---
#
class ChunkRetrieval(TransformChain):
    """Custom ChunkRetrieval Chain."""
    index: Any
    namespace: str = ""
    top_k: int = 5
    include_metadata: bool = True
    widgets: Any
    app_mode: Any
    documents: Any

    def _pinecone_chunk_retrieval(self, inputs):
        if "filter" in inputs.keys():
            filter = inputs["filter"]
            inputs.pop("filter")
        elif self.app_mode == vars.AppMode.DOCUMENTS.value:
            filter={
                "document_md5": {"$in": self.documents}
            }
        else:
            filter = None

        for key in inputs.keys():
            query = inputs[key]
            if self.widgets:
                app.expander(tab=self.widgets['tab_debug'], label=key, expanded=(key=="deixis_query"), content=query)

        if self.widgets:
            self.widgets['msg_box'].chat_message("assistant").write("Requesting query embedding")
        query_embedding = OpenAIEmbeddings(model=vars.EMBEDDING_MODEL).embed_query(query)

        if self.widgets:
            self.widgets['msg_box'].chat_message("assistant").write("Requesting matching embeddings")
        matching_embeddings = self.index.query(
            query_embedding,
            top_k=self.top_k,
            include_metadata=self.include_metadata,
            namespace=self.namespace,
            filter=filter,
        )
        return matching_embeddings.matches

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return {self.output_variables[0]: self._pinecone_chunk_retrieval(inputs)} # output "chunks"


#
# --- CHUNK FORMATTER ---
#
class ChunkFormatter(TransformChain):
    """Custom Chunk Formatter Transform Chain."""
    min_score: Any
    widgets: Any
    app_mode: Any
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

    def _format_entry_id(self, entry_id: str):
        url = entry_id.replace("arxiv", "export.arxiv")
        return url

    # --- FORMAT CHUNKS AS EXAMPLES ---
    def _format_chunks(self, chunks, app_mode):
        self.widgets['msg_box'].chat_message("assistant").write("Creating Prompt")

        # Return a array [] of dicts {"chunk": chunk} needed for the example selector
        if app_mode == vars.AppMode.DOCUMENTS.value:
            formatted_chunks = [
                f"Fuente: {db.get_document_title(item['metadata']['document_md5'])}, Página {int(item['metadata']['page'])} de {int(item['metadata']['total_pages'])}"
                f"\n\n"
                f"Relevancia: {100*item['score']:.2f}%"
                f"\n\n"
                f"Contenido: {item['metadata']['text']}"

                for item in chunks if item['score'] > self.min_score
            ]
        else:
            formatted_chunks = [
                f"Fuente: [{item['metadata']['title']}]({self._format_entry_id(item['metadata']['entry_id'])})"
                f"\n\n"
                f"Relevancia: {100*item['score']:.2f}%"
                f"\n\n"
                f"Título: {item['metadata']['text']}"

                for item in chunks if item['score'] > self.min_score
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
        formatted_chunks = self._format_chunks(inputs["chunks"], self.app_mode)
        formatted_chunks = self._few_shot_chunk_selector(formatted_chunks)
        app.expander(tab=self.widgets['tab_debug'], label="formatted_chunks", expanded=False, content=formatted_chunks)
        return {self.output_variables[0]: formatted_chunks}


#
# --- CHAT CONVERSATION ---
#
def get_chat_system_template(app_mode)  -> SystemMessagePromptTemplate:
    if app_mode == vars.AppMode.DEFAULT.value:
        source_format = '"[Título](URL)"'
    else:
        source_format = '"Título, Página X de Y"'

    system_template = """Te llamas """+vars.AI_NAME+""". Tienes muchos años de experiencia en Deep Learning.
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
        - Al final de tu respuesta, escribe la lista de Fuentes """+ source_format + """de los fragmentos que utilizaste para tu respuesta.
        """

    # Randomly address the user by its name
    if vars.username != "" and random.randint(0,10) == 0:
        system_template += f"\n\nRecuerda llamar al usuario por su nombre: {vars.username}."

    return SystemMessagePromptTemplate.from_template(template=system_template)

def create_chat_conversation(history, widgets, app_mode):
    conversation = [
            get_chat_system_template(app_mode),
        ]
    if history != None:
        conversation = conversation + history
    conversation.append(HumanMessagePromptTemplate.from_template(template="{query}"))

    app.expander(tab=widgets['tab_debug'], label="conversation_template", expanded=False, content=conversation)
    return conversation


#
# --- VERIFY CHAIN ---
#
def create_verify_prompt():
    verify_template = """
For the following given question and response, identify if the response answers to the same topic asked in the question or if it needs more information.

Question: {query}

Response: {response}

The output must be a boolean. \
Answer False if the response says that it doesnt have enough information or if it asks the user for more information or is an incompleted answer\
Answer True otherwise.
    """
    verify_prompt_template = PromptTemplate.from_template(template=verify_template)
    return verify_prompt_template


#
# --- SEARCH SEQUENTIAL CHAIN ---
#
def create_search_sequential_chain(input_variables, index, config, history, llm, widgets, documents):
    chains = []
    chains.append(ChunkRetrieval(
        transform=dummy_func,
        input_variables=["deixis_query"],
        output_variables= ["chunks"],
        index=index,
        widgets=widgets,
        namespace=config["namespace"],
        app_mode=config["app_mode"],
        documents=documents,
    ))
    chains.append(ChunkFormatter(
        transform=dummy_func,
        input_variables=["chunks"],
        output_variables=["formatted_chunks"],
        min_score=config['min_score'],
        widgets=widgets,
        app_mode=config["app_mode"],
    ))

    conversation = create_chat_conversation(history, widgets, config["app_mode"])
    chains.append(LLMChain(
            llm = llm,
            prompt=ChatPromptTemplate.from_messages(conversation), #Takes "query"
            output_key="response",
        ))

    return SequentialChain(
        chains=chains,
        input_variables=input_variables,
        output_variables=["chunks", "response"],
        # verbose=True,
    )


#
# --- GENERIC SEQUENTIAL CHAIN ---
#
def create_sequential_chain(llm, index, config, history, widgets, documents):
    chains = []

    chains.append(LLMChain(
        llm=models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass()),
        prompt=create_deixis_resolution_prompt(widgets),
        output_key="deixis_query",
    ))

    chains.append(create_search_sequential_chain(
        input_variables=["query", "deixis_query"],
        index=index,
        config=config,
        history=history,
        llm= llm,
        widgets=widgets,
        documents=documents,
    ))

    chains.append(LLMChain(
        llm = models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass()),
        prompt=create_verify_prompt(), #Takes "query" and "response"
        output_key="ai_feedback",
    ))

    return SequentialChain(
        chains=chains,
        input_variables=["query"],
        output_variables=["response", "chunks", "ai_feedback", "deixis_query"],
        # verbose=True,
    )


#
# --- SEQUENTIAL CHAIN RESPONSE ---
#
def get_chat_response(index, config, history, query, widgets, documents = []):
    MyStreamingCallbackHandler = models.MyStreamingCallbackHandlerClass()
    MyStreamingCallbackHandler.set_slow_down(widgets['slow_down'])
    MyStreamingCallbackHandler.set_widget(widgets['msg_box'])
    llm = models.get_chat_model(temperature=config['temp_slider'], handler=MyStreamingCallbackHandler)
    response = create_sequential_chain(
        llm=llm,
        index=index,
        config=config,
        history=history,
        widgets=widgets,
        documents=documents,
    )({"query":query})
    response["response"] = app.replace_urls_with_fqdn_and_lastpath(response["response"])
    return response



#
# --- CONVERSATION NAMING CHAIN ---
#
def naming_chain(query, response):
    naming_template = """
For the following given question and response, identify what the conversation about. Give special relevance to the question and complete with the response if needed.

Question: {query}

Response: {response}

The output must be a title for this conversation in the same language of the given texts.
Do not use more than 5 words for it.
    """
    naming_messages = ChatPromptTemplate.from_template(template=naming_template).format_messages(query=query, response=response)
    return models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass())(naming_messages)