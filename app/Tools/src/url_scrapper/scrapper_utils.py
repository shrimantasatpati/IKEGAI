from langchain.tools import tool
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from collections import deque
 
 
def load_embeddings():
    # embeddings=HuggingFaceEmbeddings()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint = 'https://openai-lh.openai.azure.com/',
        openai_api_key = '312ff50d6d954023b8748232617327b6',  
        deployment = 'LH-embedding',
        openai_api_version = '2023-06-01-preview',
        openai_api_type = "azure"
    )
    return embeddings
 
def llm():
    llm = AzureChatOpenAI(
            openai_api_base = "https://openai-lh.openai.azure.com/openai/deployments/LH-GPT",
            openai_api_version = '2023-06-01-preview',
            openai_api_key = '312ff50d6d954023b8748232617327b6',
            temperature = 0, max_tokens = 4096, verbose = True
        )
    return llm

def extract_text(node):
    text = []
    stack = deque([node])

    while stack:
        current_node = stack.pop()
        if current_node.string:
            text.append(current_node.string)
        else:
            children = reversed(list(current_node.children))
            stack.extend(children)
    return ''.join(text)

def get_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            content = extract_text(soup.body)
            return content
        else:
            return "failed to fetch content of url"
    except requests.exceptions.RequestException as e:
        return "error"
 
def load_vectordb(text, embeddings):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len
    ) 
    texts = text_splitter.split_text(text)
    db = Chroma.from_texts(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm(),
                                    chain_type="stuff",
                                    retriever=db.as_retriever())
    return qa
