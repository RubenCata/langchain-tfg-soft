from enum import Enum
import os
import yaml
from streamlit import cache_resource


#
# --- SET UP ENVIRONMENT VARIABLES ---
#
config = yaml.safe_load(open("../tfg-config.yml"))

openai_instance=config["openai"]["azure"]

os.environ["OPENAI_API_TYPE"] = openai_instance["api_type"]
os.environ["OPENAI_API_KEY"] = openai_instance["api_key"]
os.environ["OPENAI_API_BASE"] = openai_instance["api_base"]
os.environ["OPENAI_API_VERSION"] = openai_instance["api_version"]
os.environ["DEPLOYMENT_NAME"] = openai_instance["deployment_name"]

pinecone_instance=config["pinecone"]
os.environ["PINECONE_API_KEY"] = pinecone_instance["api_key"]
os.environ["PINECONE_ENVIRONMENT"] = pinecone_instance["environment"]

os.environ["DB_CONNECTION_STR"] = config["db"]
username = config["user"]["username"]

INDEX_NAME = pinecone_instance["index_name"]
INDEX_NAMESPACE = pinecone_instance["default_namespace"]

EMBEDDING_MODEL = "text-embedding-ada-002"
AI_NAME = os.getenv('AI_NAME', 'AI Bot')
QUERY_DEIXIS_RESOLUTION = (os.getenv('QUERY_DEIXIS_RESOLUTION', 'True').upper() == "TRUE")

MEMORY_K = 5


#
# --- ENUM CLASSES ---
#
class AppMode(Enum):
    DEFAULT = "Deep Learning"
    DOCUMENTS = "Documents"


#
# --- CONNECT TO PINECONE ---
#
@cache_resource
def get_pinecone_index():
    import pinecone
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    return pinecone.Index(INDEX_NAME)

# connect to index
def connect_to_pinecone():
    global index
    global namespace_options
    index = get_pinecone_index()
    namespace_options = sorted(list(index.describe_index_stats()["namespaces"].keys()))