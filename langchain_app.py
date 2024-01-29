from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

api_key = os.getenv('OPENAI_API_KEY')

def  ask_question(question):
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.2)

    open_ai_embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local("my_faiss_index", open_ai_embedding)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":9})

    # set prompt and create document chain
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # Doc chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    response = retrieval_chain.invoke({"input": question})
    response = str(response["answer"])
    # print(response["answer"])

    return response