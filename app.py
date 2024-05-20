import os
import warnings
import sqlite3
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from ebooklib import epub
from ebooklib.epub import EpubHtml
from bs4 import BeautifulSoup

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

# Configuration
local_llm = 'llama3'
vectorstore_dir = 'vectorstores'

if not os.path.exists(vectorstore_dir):
    os.makedirs(vectorstore_dir)

# Sanitize collection name
def sanitize_collection_name(name):
    sanitized_name = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name[:63]  # Ensure it meets the length requirement


# Enhanced read_epub function
def read_epub(file_path):
    book = epub.read_epub(file_path)
    title = None
    cover_image = None
    content = []
    author = None
    publisher = None
    publication_date = None
    language = None
    isbn = None
    subject = None

    # Extracting the title from metadata
    title_metadata = book.get_metadata('DC', 'title')
    if title_metadata:
        title = title_metadata[0][0]
    else:
        title = "Unknown Title"

    # Extracting additional metadata
    author_metadata = book.get_metadata('DC', 'creator')
    if author_metadata:
        author = author_metadata[0][0]
    
    publisher_metadata = book.get_metadata('DC', 'publisher')
    if publisher_metadata:
        publisher = publisher_metadata[0][0]

    date_metadata = book.get_metadata('DC', 'date')
    if date_metadata:
        publication_date = date_metadata[0][0]

    language_metadata = book.get_metadata('DC', 'language')
    if language_metadata:
        language = language_metadata[0][0]

    identifier_metadata = book.get_metadata('DC', 'identifier')
    if identifier_metadata:
        isbn = identifier_metadata[0][0]

    subject_metadata = book.get_metadata('DC', 'subject')
    if subject_metadata:
        subject = subject_metadata[0][0]

    # Try extracting the cover image using metadata
    cover_metadata = book.get_metadata('OPF', 'cover')
    if cover_metadata:
        cover_id = cover_metadata[0][1].get('content')
        cover_item = book.get_item_with_id(cover_id)
        if cover_item:
            cover_image = cover_item.get_content()

    # Fallback to checking if 'cover' is in the image name
    if not cover_image:
        for item in book.get_items_of_type(epub.ITEM_IMAGE):
            if 'cover' in item.get_name().lower():
                cover_image = item.get_content()

    # Extract content from document items
    for item in book.get_items():
        if isinstance(item, EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            content.append(soup.get_text())

    return title, cover_image, content, author, publisher, publication_date, language, isbn, subject


   # Debugging prints
    print(f"Title extracted: {title}")
    #print(f"Cover image extracted: {'Yes' if cover_image else 'No'}")
    #print(f"Number of documents extracted: {len(content)}")
    print(f"Author extracted: {author}")
    print(f"Publisher extracted: {publisher}")
    print(f"Publication date extracted: {publication_date}")
    print(f"Language extracted: {language}")
    print(f"ISBN extracted: {isbn}")
    print(f"Subject extracted: {subject}")



# Setup SQLite database
def setup_database(db_path='books.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY,
        title TEXT,
        cover_image BLOB,
        author TEXT,
        publisher TEXT,
        publication_date TEXT,
        language TEXT,
        isbn TEXT,
        subject TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS texts (
        id INTEGER PRIMARY KEY,
        book_id INTEGER,
        content TEXT,
        FOREIGN KEY (book_id) REFERENCES books (id)
    )
    ''')
    
    conn.commit()
    return conn

def book_exists(conn, title):
    c = conn.cursor()
    c.execute('SELECT id FROM books WHERE title = ?', (title,))
    return c.fetchone() is not None

def insert_book(conn, title, cover_image, author, publisher, publication_date, language, isbn, subject):
    c = conn.cursor()
    c.execute('''
    INSERT INTO books (title, cover_image, author, publisher, publication_date, language, isbn, subject) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, cover_image, author, publisher, publication_date, language, isbn, subject))
    conn.commit()
    return c.lastrowid

def insert_texts(conn, book_id, texts):
    c = conn.cursor()
    c.executemany('INSERT INTO texts (book_id, content) VALUES (?, ?)', [(book_id, text) for text in texts])
    conn.commit()

def save_vectorstore(doc_splits, embeddings, book_key):
    data = {
        'doc_splits': doc_splits,
        'embeddings': embeddings
    }
    with open(os.path.join(vectorstore_dir, f'{book_key}.pkl'), 'wb') as file:
        pickle.dump(data, file)

def load_vectorstore(book_key):
    vectorstore_path = os.path.join(vectorstore_dir, f'{book_key}.pkl')
    if os.path.exists(vectorstore_path):
        try:
            with open(vectorstore_path, 'rb') as file:
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            print(f"Error loading vectorstore for book '{book_key}'. Recreating vectorstore.")
            os.remove(vectorstore_path)
    return None

# Path to the EPUB file
epub_file_path = 'books/jony-ive.epub'
if not os.path.exists(epub_file_path):
    print("EPUB file not found.")
    exit(1)

print("Loading and indexing documents from EPUB...")
title, cover_image, epub_content, author, publisher, publication_date, language, isbn, subject = read_epub(epub_file_path)


# Generate a unique key for the book
book_key = sanitize_collection_name(title.replace(" ", "_").lower())

# Validate extracted data
if title is None or cover_image is None or not epub_content:
    print("Failed to extract title, cover image, or text content from EPUB.")
    exit(1)

docs_list = [Document(page_content=content) for content in epub_content]

# Split text into manageable parts
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Setup and populate the database
conn = setup_database()

if not book_exists(conn, title):
    book_id = insert_book(conn, title, cover_image, author, publisher, publication_date, language, isbn, subject)
    insert_texts(conn, book_id, [doc.page_content for doc in doc_splits])
else:
    print(f"Book '{title}' already exists in the database.")

# Check if vectorstore exists for this book
vectorstore_data = load_vectorstore(book_key)
if vectorstore_data is not None:
    print(f"Loading vectorstore for book '{title}' from pickle file...")
    doc_splits = vectorstore_data.get('doc_splits', [])
    embeddings = vectorstore_data.get('embeddings', [])
    if doc_splits and embeddings:
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=sanitize_collection_name(book_key),  # Use sanitized book key
            embedding=embeddings
        )
    else:
        print(f"Missing data in pickle file for book '{title}'. Recreating vectorstore.")
        vectorstore = None
else:
    vectorstore = None

if vectorstore is None:
    # Index documents in a vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=sanitize_collection_name(book_key),  # Use sanitized book key
        embedding=GPT4AllEmbeddings(),
    )
    # Save the vectorstore data to a pickle file
    save_vectorstore(doc_splits, GPT4AllEmbeddings(), sanitize_collection_name(book_key))




retriever = vectorstore.as_retriever()

# Setup LangChain prompt for answer generation
prompt_template = PromptTemplate(
    template="Try to use only the provided context and make it clear when your answer is not comming from the book.: {context}, answer the following question: {question}",
    input_variables=["context", "question"],
)

llm = ChatOllama(model=local_llm)
answer_generator = prompt_template | llm | StrOutputParser()

# Interaction with user in the terminal
while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    retrieved_docs = retriever.invoke(question, num_results=10)  # Retrieve multiple documents

    if retrieved_docs:
        # Combine contexts from multiple documents
        combined_context = " ".join([doc.page_content for doc in retrieved_docs])
        print(f"Combined context: {combined_context[:500]}... \n")  # Show a snippet of the combined context