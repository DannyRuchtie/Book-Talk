from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import warnings

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

# Configuration
local_llm = 'llama3'

# Function to read and extract text from an EPUB file
def read_epub(file_path):
    book = epub.read_epub(file_path)
    content = []
    for item in book.get_items():
        if item.get_type() == 9:  # 9 corresponds to ebooklib.ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            content.append(soup.get_text())
    return content

# Path to the EPUB file
epub_file_path = 'books/book.epub'
if not os.path.exists(epub_file_path):
    print("EPUB file not found.")
    exit(1)

print("Loading and indexing documents from EPUB...")
epub_content = read_epub(epub_file_path)
docs_list = [Document(page_content=content) for content in epub_content]

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
while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break

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