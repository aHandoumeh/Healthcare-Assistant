from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import re

# extract from urls: 
with open("urls.txt", "r", encoding="utf-8") as file:
    text = file.read()

urls = re.findall(r'https?://[^\s<>"\']+', text)
urls = [url.rstrip(".,;:!?)]}") for url in urls]


# Load from webpages:
loader = WebBaseLoader(urls)
documents = loader.load()

# Clean
for doc in documents:
    doc.page_content = re.sub(
        r'\s+',
        ' ',
        doc.page_content
    ).strip()

print('*' * 50)

# Chunking:
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Load to database:
chunks = text_splitter.split_documents(documents)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)


