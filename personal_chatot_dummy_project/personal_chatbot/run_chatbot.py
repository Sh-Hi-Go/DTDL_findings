import os
import logging
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import chainlit as cl
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
"""
def load_model():
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024, 
                    temperature=0.2, top_p=0.95, repetition_penalty=1.15, do_sample=True)

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm
"""

def main():
    # load embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"})

    # load datastore
    db = Chroma(persist_directory = PERSIST_DIRECTORY, embedding_function = embeddings, client_settings= CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # load LLM
    #llm = load_model()
    #print(type(llm))
    
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)#, return_source_documents=True)
    qa = RetrievalQA.from_chain_type(chain_type="stuff", retriever=retriever)#, return_source_documents=True)
    print(type(qa))

    query = input("\nENTER A QUERY: ")

        
    response = qa(query)
    print(type(response))
    #answer = response['result']
    #docs = response['source_documents']
    
    #print("\n\n> Question: ")
    #print(query)
    #print("\n> Answer: ")
    #print(answer)

    #print("------------------SOURCE DOCUMENTS------------------")
    #for document in docs:
    #    print("\n> " + document.metadata["source"] + ":")
    #    print(document.page_content)
        
    #print("------------------SOURCE DOCUMENTS------------------")


    """
    # QnA
    while True:
        query = input("\nENTER A QUERY: ")

        if query=="exit":
            break

        response = qa(query)
        answer, docs = response['result'], response['source_documents']

        print("\n\n> Question: ")
        print(query)
        print("\n> Answer: ")
        print(answer)

        print("------------------SOURCE DOCUMENTS------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        
        print("------------------SOURCE DOCUMENTS------------------")
"""
if __name__ == "__main__":
    main()

"""
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"})

db = Chroma(persist_directory = PERSIST_DIRECTORY, embedding_function = embeddings, client_settings= CHROMA_SETTINGS)
retriever = db.as_retriever()

llm = load_model()

@cl.on_chat_start
def main():
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=retriever)
    cl.user_session.set("retrieval_chain", retrieval_chain)
    
@cl.on_message
async def main(message:str):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    callback = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    callback.answer_reached = True
    res = await retrieval_chain.acall(message, callbacks=[callback])

    await cl.Message(content=res["result"]).send()
"""