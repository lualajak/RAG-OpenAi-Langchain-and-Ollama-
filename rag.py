import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma

load_dotenv()

url = "https://mojca.gov.ss/constitution-and-government"
embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")


llm = ChatOpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    model = "deepseek/deepseek-chat:free",
    base_url=os.getenv("BASE_URL")

)


loader = WebBaseLoader(url)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    collection_name="simple-rag"
)

QUERY_PROMPT = PromptTemplate(
    input_variables=['question'],
    template= """ You are language assistant. your task is generate two other questions 
    from the question given by the user. original question: {question}"""
)

# retriever = vectordb.as_retriever()
retriever = MultiQueryRetriever.from_llm(
    vectordb.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

template = """ Answer the questions ONLY base of  context: {context}
question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = ({"context":retriever, "question":RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
)

# response = chain.invoke("what is the document about")
# response = chain.invoke(input=("what is the document about"))

response = chain.invoke(input=("summaries the document"))
print(response)

    

