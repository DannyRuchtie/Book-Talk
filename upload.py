import os
import sqlite3
import pickle
import warnings
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.schema import Document
from ebooklib import epub
from ebooklib.epub import EpubHtml
from bs4 import BeautifulSoup

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

ALLOWED_EXTENSIONS = {'epub', 'epub.zip'}

def get_db_connection():
    conn = sqlite3.connect('books.db')
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    print(f"Checking file: {filename}")  # Debug statement
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    title_metadata = book.get_metadata('DC', 'title')
    if title_metadata:
        title = title_metadata[0][0]
    else:
        title = "Unknown Title"

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

    cover_metadata = book.get_metadata('OPF', 'cover')
    if cover_metadata:
        cover_id = cover_metadata[0][1].get('content')
        cover_item = book.get_item_with_id(cover_id)
        if cover_item:
            cover_image = cover_item.get_content()

    if not cover_image:
        for item in book.get_items_of_type(epub.ITEM_IMAGE):
            if 'cover' in item.get_name().lower():
                cover_image = item.get_content()

    for item in book.get_items():
        if isinstance(item, EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            content.append(soup.get_text())

    return title, cover_image, content, author, publisher, publication_date, language, isbn, subject

def process_epub(file_path):
    title, cover_image, epub_content, author, publisher, publication_date, language, isbn, subject = read_epub(file_path)

    print(f"Processing EPUB file: {file_path}")
    print(f"Title: {title}, Author: {author}, Publisher: {publisher}")

    if not title or not epub_content:
        print("Failed to extract title or content from EPUB.")
        return

    book_key = title.replace(" ", "_").lower()
    docs_list = [Document(page_content=content) for content in epub_content]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    conn = get_db_connection()
    if not book_exists(conn, title):
        book_id = insert_book(conn, title, cover_image, author, publisher, publication_date, language, isbn, subject)
        insert_texts(conn, book_id, [doc.page_content for doc in doc_splits])
        print(f"Inserted book into database with ID: {book_id}")
    conn.close()

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="simple-chroma",
        embedding=GPT4AllEmbeddings(),
    )
    save_vectorstore(doc_splits, GPT4AllEmbeddings(), book_key)
    print(f"Vectorstore created and saved for book key: {book_key}")

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
    with open(os.path.join('vectorstores', f'{book_key}.pkl'), 'wb') as file:
        pickle.dump(data, file)

def load_vectorstore(book_key):
    vectorstore_path = os.path.join('vectorstores', f'{book_key}.pkl')
    if os.path.exists(vectorstore_path):
        try:
            with open(vectorstore_path, 'rb') as file:
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            print(f"Error loading vectorstore for book '{book_key}'. Recreating vectorstore.")
            os.remove(vectorstore_path)
    return None
