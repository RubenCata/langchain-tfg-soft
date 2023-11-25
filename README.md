# TFG Software - Rubén Catalán

## Descripción

  En este repositorio ha desarrolado un bot conversacional con el que uno pueda preguntar sobre su propia información de forma privada.
  Para ello, principalmente se ha usado Python, Langchain, Streamlit, Pinecone, MySQL y un Endpoint de OpenAI en Azure 
  
  Esta aplicación se ha desarrollado sobre un caso de uso de ejemplo usando como fuente de información 3500 artículos sobre Deep Learning de la página web de arXiv.
  Estos documentos representan la fuente de información privada que el usuario debe indexar previamente en Pinecone dentro de un namespace para hacer preguntas sobre ella.
  Además, se añade una opción en la que el usuario pueda cargar sus propios documentos para posteriormente hacer consultas sobre ellos.

## Claves y datos propios de cada usuario

  Esta aplicación usa servicios para los que se necesitarán claves y datos que son únicos para cada usuario.
  Estos datos se deben incorporar en un archivo ```tfg-config.yml``` en el directorio donde se clone este repositorio de Git.

  El archivo ```tfg-config.yml``` debe tener los siguientes datos con exactamente la misma estructura:
  
  ```
  openai:
    azure:
      api_type: "azure"
      api_key: "<YOUR_API_KEY>"
      api_version: "<YOUR_API_VERSION>"
      api_base: "https://<YOUR_ENDPOINT>.openai.azure.com/"
      deployment_name: "<YOUR_MODEL_DEPLOYMENT>"

  pinecone:
    api_key: "<YOUR_API_KEY>"
    environment: "<YOUR_INDEX_ENVIRONMENT>"
    index_name: "<YOUR_INDEX_NAME>"
    default_namespace: "<YOUR_INDEX_NAMESPACE>"

  db: "mysql+pymysql://<MYSQL_USER>:<MYSQL_PASSWORD>@localhost:<YOUR_DB_PORT>/<YOUR_DB_NAME>"

  user:
    username: "<YOUR_NAME>"
  ```

## Intrucciones de ejecución

  Para la aplicación se ha usado:
  - Python ```3.10.11```
  - MySQL ```8.0```
  - OpenAI Endpoint en Azure
  - Base de datos en Pinecone
  - Librerías especificadas en el archivo ```requirements.txt```

  Para clonar el repositorio de Git, se debe de usar la siguiente sentencia:
  
  ```$ git clone https://github.com/RubenCata/langchain-tfg-soft```

  El proyecto requiere una base de datos MySQL disponible en localhost con el nombre, usuario y contraseña que se indiquen en el archivo propio ```tfg-config.yml```

  También se requiere un index en una base de datos de vectores de Pinecone con el nombre, namespace y entorno que se indiquen en el archivo propio ```tfg-config.yml```
  Es importante tener previamente indexada la información privada en un namespace especifico. En caso de no tener información privada, se deberá crear un namespace vacío con el que trabajar por defecto.

  Aunque no es necesario, si es recomendable crearse un entorno virtual en el que ejecutar la instalar las librerias y ejecutar la aplicación. Se puede saltar este paso pero podría haber conflictos de librerias si tienes otros proyectos de Python en tu ordenador. La creación y activación del entorno virtual sería  usando:

  ```$ python -m venv venv```
  
  ```$ venv\Scripts\activate```
  
  Para instalar las librerias necesarias e indicadas en el archivo ```requirements.txt```, se debe usar la siguiente sentencia:

  ```$ pip install -r requirements.txt```

  Una vez instalado todo lo necesario, la aplicación se puede ejecutar en local simplemente con la siguiente sentencia:

  ```$ streamlit run app\main.py```
