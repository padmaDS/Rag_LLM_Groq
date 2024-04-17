import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize embeddings
embeddings = OllamaEmbeddings()

# Initialize document loader
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

print("documents are" , docs)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])

print("final documents are" , final_documents)

# Create vectors from documents
vectors = FAISS.from_documents(final_documents, embeddings)
print("vectors are" , vectors)

# Initialize ChatGroq with API key and model name
llm = ChatGroq(temperature=0,
               groq_api_key=groq_api_key, 
               model_name="mixtral-8x7b-32768")

# Define ChatPromptTemplate
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Example query
query = "what is langsmith?"

# Invoke the retrieval chain with the query
response = retrieval_chain.invoke({"input": query})

# Print or use the response
print(response['answer'])

# Now you can use retrieval_chain to invoke queries and retrieve responses
