from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Configuration
local_llm = 'llama3'

# Load documents from URLs
urls = [
    "https://developer.apple.com/design/human-interface-guidelines/designing-for-visionos",
    "https://developer.apple.com/design/human-interface-guidelines/designing-for-ios",
    "https://developer.apple.com/design/human-interface-guidelines/designing-for-ipados",
]

print("Loading and indexing documents...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split text into manageable parts
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Index documents in a vector database
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="simple-chroma",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Setup LangChain prompt for answer generation
prompt_template = PromptTemplate(
    template="Given the context: {context}, answer the question: {question}",
    input_variables=["context", "question"],
)

llm = ChatOllama(model=local_llm)
answer_generator = prompt_template | llm | StrOutputParser()

# Interaction with user in the terminal
question = input("Enter your question: ")
retrieved_docs = retriever.invoke(question)

if retrieved_docs:
    # Retrieve the first relevant document
    context = retrieved_docs[0].page_content
    print(f"Retrieved document content: {context[:500]}...")  # Show a snippet of the document

    # Generate an answer based on the retrieved document
    answer = answer_generator.invoke({"context": context, "question": question})
    print(f"Answer: {answer}")
else:
    print("No relevant documents were found for your question.")

