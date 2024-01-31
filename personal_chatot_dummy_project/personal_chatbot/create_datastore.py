import os
import logging
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, ROOT_DIRECTORY

def load_single_document(file_path:str) -> Document:
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding = "utf8")

    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)

    return loader.load()[0]


def load_documents(source_dir: str) -> list[Document]:
    all_files = os.listdir(source_dir)
    x= [f"{source_dir}/{file_path}" for file_path in all_files if file_path[-4:] in ['.txt','.pdf','.docx', '.csv']]
    y=[]

    for i in x:
        y.append(load_single_document(i))
    
    return y

def main():
    # load documents
    documents = load_documents(SOURCE_DIRECTORY)
    
    # split in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"})

    # store them as knowledge base
    db = Chroma.from_documents(texts, embeddings, persist_directory = PERSIST_DIRECTORY, client_settings= CHROMA_SETTINGS)
    db.persist()
    db=None

if __name__ == "__main__":
    main()