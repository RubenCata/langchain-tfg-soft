import json
import os
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

import app_langchain.models as models


def dummy_func(inputs: dict) -> dict:
    return inputs


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
        llm = models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass()),
        prompt=ChatPromptTemplate.from_template(query_classifier_prompt_template),
        verbose=False,
    )


#
# --- PINECONE CHUNK RETRIEVAL ---
#
def pinecone_chunk_retrieval(inputs, index, namespace, top_k, include_metadata, widgets):
    if "filter" in inputs.keys():
        filter = inputs["filter"]
        inputs.pop("filter")
    else:
        filter = None

    label = ""
    for key in inputs.keys():
        query = inputs[key]
        app.expander(tab=widgets['tab_debug'], label=key, expanded=(key=="deixis_query"), content=query)

    widgets['msg_box'].chat_message("assistant").write("Requesting query embedding")
    query_embedding = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"]).embed_query(query)

    widgets['msg_box'].chat_message("assistant").write("Requesting matching embeddings")
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
    widgets: Any

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return {self.output_variables[0]: self.transform(inputs, self.index, self.namespace, self.top_k, self.include_metadata, self.widgets)}


#
# --- FOCUS CHAIN ---
#
def focus_func(inputs: dict, index, namespace, history, widgets) -> dict:
    widgets['msg_box'].chat_message("assistant").write("Filtering results")
    filter=None
    keywords = {
    }
    for key in inputs.keys():
        query = inputs[key]

    output_focus_chain = query_classifier_chain().run({"query": query,"history": history}).replace("`", "")
    app.expander(tab=widgets['tab_debug'], label="output_focus_chain", expanded=False, content=output_focus_chain)

    try:
        qclass = json.loads(output_focus_chain)
    except json.JSONDecodeError as e:
        qclass = {}

    app.expander(tab=widgets['tab_debug'], label="qclass", expanded=True, content=qclass )

    if "Type" in qclass.keys() and qclass["Type"] in keywords.keys():
        focus_query = keywords[qclass["Type"]] + ", " + qclass["Name"]
        vect = ChunkRetrieval(
            transform=pinecone_chunk_retrieval,
            input_variables=["focus_query"],
            output_variables=["focus_chunk"],
            index=index,
            namespace=namespace,
            top_k=1,
            widgets=widgets
        )({"focus_query": focus_query})["focus_chunk"]
        filter = { "title": vect[0].metadata['title'] }

    app.expander(tab=widgets['tab_debug'], label="filter", expanded=True, content=filter )

    return {"filter": filter}


class FocusChain(TransformChain):
    """Custom FocusChain."""
    index: Any
    namespace: str = ""
    history: Any
    widgets: Any

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return self.transform(inputs, self.index, self.namespace, self.history, self.widgets)


#
# --- CHAT CONVERSATION ---
#
def get_chat_system_template()  -> SystemMessagePromptTemplate:
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
- Al final de tu respuesta, escribe la lista de Fuentes [Título](URL) de los fragmentos que utilizaste para tu respuesta.
"""
# Glosario:
# -

    # Randomly address the user by its name
    if vars.username != "" and random.randint(0,10) == 0:
        system_template += f"\n\nRecuerda llamar al usuario por su nombre: {vars.username}."

    return SystemMessagePromptTemplate.from_template(template=system_template)

def create_chat_conversation(history, widgets):
    conversation = [
            get_chat_system_template(),
        ]
    if history != None:
        conversation = conversation + history[1:]
    conversation.append(HumanMessagePromptTemplate.from_template(template="{query}"))

    app.expander(tab=widgets['tab_debug'], label="conversation_template", expanded=False, content=conversation)
    return conversation


#
# --- CHUNK FORMATTER ---
#
class ChunkFormatter(TransformChain):
    """Custom Chunk Formatter Transform Chain."""
    min_score: Any
    widgets: Any
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
        self.widgets['msg_box'].chat_message("assistant").write("Creating Prompt")

        # Return a array [] of dicts {"chunk": chunk} needed for the example selector
        formatted_chunks = [
            f"Fuente: [{item['metadata']['title']}]({item['metadata']['url']})"
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
        formatted_chunks = self._format_chunks(inputs["chunks"])
        formatted_chunks = self._few_shot_chunk_selector(formatted_chunks)
        app.expander(tab=self.widgets['tab_debug'], label="formatted_chunks", expanded=False, content=formatted_chunks)
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
def create_search_sequential_chain(input_variables, index, config, history, llm, widgets):
    chains = []
    chains.append(ChunkRetrieval(
        transform=pinecone_chunk_retrieval,
        input_variables=["deixis_query", "filter"] if "filter" in input_variables else ["deixis_query"],
        output_variables= ["chunks"],
        index=index,
        widgets=widgets,
        namespace=config["namespace"],
    ))
    chains.append(ChunkFormatter(
        transform=dummy_func,
        min_score=config['min_score'],
        widgets=widgets,
        input_variables=["chunks"],
        output_variables=["formatted_chunks"],
    ))

    conversation = create_chat_conversation(history, widgets)
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
def create_sequential_chain(llm, index, config, history, focus_chain_on, widgets):
    chains = []

    chains.append(TransformChain(
        transform=query_transform_func,
        input_variables=["initial_query"],
        output_variables=["query"],
    ))
    if config["query_deixis_resolution"]:
        chains.append(LLMChain(
            llm=models.get_chat_model(handler=models.FakeStreamingCallbackHandlerClass()),
            prompt=create_deixis_resolution_prompt(widgets),
            output_key="deixis_query",
        ))
    else:
        chains.append(TransformChain(
            transform=dummy_func,
            input_variables=["query"],
            output_variables=["deixis_query"],
        ))
    # Find a metadata filter, only apply to Confluence namespaces
    if focus_chain_on and ("ddo" in config["namespace"]):
        output_variables = ["response", "filter"]
        chains.append(FocusChain(
            transform=focus_func,
            input_variables=["deixis_query"],
            output_variables=["filter"],
            index=index,
            namespace=config["namespace"],
            history=history,
            widgets=widgets
        ))
        chains.append(create_search_sequential_chain(
            input_variables=["query", "deixis_query", "filter"],
            index=index,
            config=config,
            history=history,
            llm= llm,
            widgets=widgets,
        ))
    else:
        output_variables=["response"]
        chains.append(create_search_sequential_chain(
            input_variables=["query", "deixis_query"],
            index=index,
            config=config,
            history=history,
            llm= llm,
            widgets=widgets,
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
def get_chat_response(index, config, history, focus_chain_on, query, widgets):
    MyStreamingCallbackHandler = models.MyStreamingCallbackHandlerClass()
    MyStreamingCallbackHandler.set_slow_down(widgets['slow_down'])
    MyStreamingCallbackHandler.set_widget(widgets['msg_box'])
    llm = models.get_chat_model(temperature=config['temp_slider'], handler=MyStreamingCallbackHandler)
    response = create_sequential_chain(
        llm=llm,
        index=index,
        config=config,
        history=history,
        focus_chain_on=focus_chain_on,
        widgets=widgets,
    )({"initial_query":query})
    response["response"] = app.replace_urls_with_fqdn_and_lastpath(response["response"])
    return response