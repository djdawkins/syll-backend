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

    # Load pdf and initiate
    loader = PyPDFLoader("FA22_NURS-112 Sec 81 - Prof Prac Suc I sp 2024.pdf")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docs = loader.load()

    # Split text and create vector
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    # set prompt and create document chain
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and format the anwser like a blog post:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    quiz_questions = ["When and where is the course meeting?", 
                  "What are the required textbooks for the course?",
                  "How will your final grade be calculated?",
                  "What is the instructor's policy on late assignments?",
                  "What is the attendance policy for the course?",
                  "How can you access additional course materials and resources?",
                  "What is the instructor's preferred method of communication?"]

    response = retrieval_chain.invoke({"input": question})
    response = str(response["answer"]).strip()
    # print(response["answer"])

    return response