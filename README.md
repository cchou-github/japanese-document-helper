# Japanese Documentation Helper
## Introduction
A Japanese documentation helper created using open-source embeddings model and LLM.

The chosen open-source models have excellent compatibility with Japanese. Since all models used are open-source, there is no need to call any APIs during execution, allowing everything to be executed locally. This significantly reduces the risk of leaking confidential data.

A lightweight streamlit-chat frontend is used to create a chatbot to answer any questions you have about the documentation.

## About the models
* Embedding model: intfloat/multilingual-e5-large
* LLM model: elyza/ELYZA-japanese-Llama-2-7b-instruct  

The models will be dowloaded automatically on the first run and cached in `~/.cache/huggingface/hub`. Each model will be nearly 10GB.

## Enviroment
* Python 11
* Langchain 0.1.11
* ChromaDB 0.4.24
* streamlit-chat = "0.1.1"

## Set up
To set up, use Pipenv: `pipenv install`

## Usage
First, activate the Pipenv environment:`pipenv shell`

### Embeddings
You must prepare HTML files and embed them.

`ingestions.py` is an example of embedding, which stores the embedded vector data in ChromaDB.
You can easily scrape HTML files using the following wget command:
`wget -r -P source-docs <the starting URL that you want to scrap>`

Next, run `python ingestions.py`. This program will automatically embed the HTML files in the source-docs directory.

### Start the Chatbot
Start the server: `streamlit run main.py`  
The server runs in port 8501, so the URL is http://localhost:8501.