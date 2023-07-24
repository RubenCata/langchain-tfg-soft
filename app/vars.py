import os
import requests
import base64
import json
import yaml
import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers



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